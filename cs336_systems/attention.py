import torch
import torch.autograd as autograd
import torch.nn.functional as F
import triton
import triton.language as tl


class FlashAttentionAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,causual:bool,dropout_p:float = 0.0):
        d_k = q.size(-1)
        scale = 1 / (d_k ** 0.5)
        attn_scores = q @ k.transpose(-1,-2) * scale #b,h,s,s
        seq_len = attn_scores.size(-1)
        if causual:
            mask = torch.triu(torch.ones(seq_len,seq_len,device = q.device),diagonal = 1).bool()
            attn_scores = attn_scores.masked_fill(mask,-torch.inf)
        attn_weight = torch.softmax(attn_scores,dim = -1)
        o = attn_weight @ v
        lse = torch.logsumexp(attn_scores, dim=-1)
        
        ctx.save_for_backward(q,k,v,attn_weight,lse)
        ctx.causual = causual
        ctx.scale = scale #反向传播也要scale
        
        return o
    
    @staticmethod    
    def backward(ctx,dO):
        q,k,v,attn_weight,lse = ctx.saved_tensors
        causual = ctx.causual
        scale = ctx.scale
        
        A = attn_weight
        dV = attn_weight.transpose(-1,-2) @ dO 
        dA = dO @ v.transpose(-1, -2)
        
        sum_term = (dA * A).sum(dim=-1, keepdim=True)
        dS = A * (dA - sum_term)

        d_Qkt = scale * dS
        dQ = d_Qkt @ k
        dK = d_Qkt.transpose(-1,-2) @ q
        
        if causual:
            seq_len = dS.size(-1)
            mask = torch.triu(torch.ones(seq_len,seq_len,device = q.device),diagonal = 1).bool()
            dQ.masked_fill(mask,0.0)
            dK.masked_fill(mask,0.0)
        
        return dQ,dK,dV,None,None

@triton.jit
def _triton_flash_forward_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, l_ptr,
    B, Tq, Tk, D, scale, is_causal,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    b = pid_b

    q_block_start = pid_q * BLOCK_Q
    offs_q = q_block_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q
    q_base = b * Tq * D + offs_q[:, None] * D + offs_d[None, :]
    Q = tl.load(q_ptr + q_base, mask=(offs_q[:, None] < Tq) & (offs_d[None, :] < D), other=0.0)

    # Initialize running statistics
    m_i = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
    O_i = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    for k_start in range(0, Tk, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Load K and V
        k_base = b * Tk * D + offs_k[:, None] * D + offs_d[None, :]
        K = tl.load(k_ptr + k_base, mask=(offs_k[:, None] < Tk) & (offs_d[None, :] < D), other=0.0)
        v_base = b * Tk * D + offs_k[:, None] * D + offs_d[None, :]
        V = tl.load(v_ptr + v_base, mask=(offs_k[:, None] < Tk) & (offs_d[None, :] < D), other=0.0)

        # Compute scores
        score = scale * tl.dot(Q, tl.trans(K))
        
        # Apply causal mask if needed
        if is_causal:
            # Query i can attend to key j where j <= i
            # Mask out j > i (future tokens)
            mask = offs_k[None, :] > offs_q[:, None]
            score = tl.where(mask, -float("inf"), score)

        # For each row, check if there is at least one valid key
        # A row is valid if max(score) > -inf/2 (i.e., not -inf)
        m_ij = tl.max(score, axis=1)
        row_has_valid = m_ij > -float("inf") / 2
        
        # For rows with no valid keys, we want to skip the update
        # We do this by:
        # 1. For invalid rows, set m_ij to m_i (so m_new = m_i, no change)
        # 2. For invalid rows, set l_ij to 0 (so no contribution to l_i)
        # 3. For invalid rows, set p_ij to 0 (so no contribution to O_i)
        
        # Compute p_ij and l_ij for all rows
        # For invalid rows, this would be exp(-inf - (-inf)) = NaN
        # So we compute it safely by using a mask
        p_ij = tl.where(row_has_valid[:, None], tl.exp(score - m_ij[:, None]), 0.0)
        l_ij = tl.sum(p_ij, axis=1)
        
        # For invalid rows, keep old values
        m_ij = tl.where(row_has_valid, m_ij, m_i)
        l_ij = tl.where(row_has_valid, l_ij, 0.0)
        
        # Update running statistics
        m_new = tl.maximum(m_i, m_ij)
        
        # Compute scale factors
        # For invalid rows: m_i == m_ij == m_new, so alpha = beta = 1.0
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        # For invalid rows: p_ij = 0, l_ij = 0, so no contribution
        O_i = O_i * alpha[:, None] + tl.dot(p_ij, V) * beta[:, None]
        l_i = l_i * alpha + l_ij * beta
        m_i = m_new

    # Final log-sum-exp and normalization
    l_out = tl.log(l_i) + m_i
    O_i = O_i / l_i[:, None]
    
    # Store output
    out_base = b * Tq * D + offs_q[:, None] * D + offs_d[None, :]
    tl.store(out_ptr + out_base, O_i, mask=(offs_q[:, None] < Tq) & (offs_d[None, :] < D))
    
    # Store log-sum-exp
    l_base = b * Tq + offs_q
    tl.store(l_ptr + l_base, l_out, mask=offs_q < Tq)

@triton.jit
def _triton_flash_backward_kernel(
    dout_ptr, q_ptr, k_ptr, v_ptr, l_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    B, Tq, Tk, D, scale, is_causal,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_k = tl.program_id(0)
    pid_b = tl.program_id(1)
    b = pid_b

    k_block_start = pid_k * BLOCK_K
    offs_k = k_block_start + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    # Load K and V
    k_base = b * Tk * D + offs_k[:, None] * D + offs_d[None, :]
    K = tl.load(k_ptr + k_base, mask=(offs_k[:, None] < Tk) & (offs_d[None, :] < D), other=0.0)
    v_base = b * Tk * D + offs_k[:, None] * D + offs_d[None, :]
    V = tl.load(v_ptr + v_base, mask=(offs_k[:, None] < Tk) & (offs_d[None, :] < D), other=0.0)

    dK = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    dV = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    for q_start in range(0, Tq, BLOCK_Q):
        offs_q = q_start + tl.arange(0, BLOCK_Q)
        
        # Load Q, dO, and L
        q_base = b * Tq * D + offs_q[:, None] * D + offs_d[None, :]
        Q = tl.load(q_ptr + q_base, mask=(offs_q[:, None] < Tq) & (offs_d[None, :] < D), other=0.0)
        dout_base = b * Tq * D + offs_q[:, None] * D + offs_d[None, :]
        DOUT = tl.load(dout_ptr + dout_base, mask=(offs_q[:, None] < Tq) & (offs_d[None, :] < D), other=0.0)
        l_base = b * Tq + offs_q
        L = tl.load(l_ptr + l_base, mask=offs_q < Tq, other=0.0)

        # Compute scores
        score = scale * tl.dot(Q, tl.trans(K))
        
        # Apply causal mask
        if is_causal:
            mask = offs_k[None, :] > offs_q[:, None]
            score = tl.where(mask, -float("inf"), score)
        
        # P = softmax(score) = exp(score - L)
        # For rows where all scores are -inf, L is -inf, so exp(-inf - (-inf)) = NaN
        # But L is stored as log-sum-exp, and for fully masked rows, L = -inf (or very negative)
        # Actually, L is stored per query position, and for causal attention,
        # L should be finite for all positions that have at least one valid key
        # For positions with no valid keys (shouldn't happen in valid causal attention),
        # L would be -inf, but we handle it by making P = 0
        
        # Check if row has any valid key
        row_has_valid = tl.max(score, axis=1) > -float("inf") / 2
        
        # Compute P safely: for invalid rows, set P to 0
        P = tl.where(row_has_valid[:, None], tl.exp(score - L[:, None]), 0.0)
        
        # Gradients - these will naturally be zero where P is 0
        dV += tl.dot(tl.trans(P), DOUT)
        
        dP = tl.dot(DOUT, tl.trans(V))
        dS = P * (dP - tl.sum(P * dP, axis=1)[:, None])
        
        dK += scale * tl.dot(tl.trans(dS), Q)
        
        dQ_local = scale * tl.dot(dS, K)
        dq_base = b * Tq * D + offs_q[:, None] * D + offs_d[None, :]
        tl.atomic_add(dq_ptr + dq_base, dQ_local, mask=(offs_q[:, None] < Tq) & (offs_d[None, :] < D))

    tl.atomic_add(dk_ptr + k_base, dK, mask=(offs_k[:, None] < Tk) & (offs_d[None, :] < D))
    tl.atomic_add(dv_ptr + v_base, dV, mask=(offs_k[:, None] < Tk) & (offs_d[None, :] < D))

class FlashAttentionAutogradFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        B, Tq, D = q.shape
        Tk = k.shape[1]
        scale = 1.0 / (D ** 0.5)

        BLOCK_Q = 64
        BLOCK_K = 64
        BLOCK_D = 64
        
        BLOCK_Q = min(BLOCK_Q, Tq)
        BLOCK_K = min(BLOCK_K, Tk)
        BLOCK_D = min(BLOCK_D, D)
        
        out = torch.empty_like(q)
        L = torch.empty((B, Tq), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(Tq, BLOCK_Q), B)
        
        _triton_flash_forward_kernel[grid](
            q, k, v, out, L,
            B, Tq, Tk, D, scale, is_causal,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
        )

        ctx.save_for_backward(q, k, v, L)
        ctx.scale = scale
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, L = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        B, Tq, D = q.shape
        Tk = k.shape[1]

        BLOCK_Q = min(64, Tq)
        BLOCK_K = min(64, Tk)
        BLOCK_D = min(64, D)
        
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        grid = (triton.cdiv(Tk, BLOCK_K), B)
        
        _triton_flash_backward_kernel[grid](
            dout, q, k, v, L,
            dq, dk, dv,
            B, Tq, Tk, D, scale, is_causal,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
        )
        return dq, dk, dv, None