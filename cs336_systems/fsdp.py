import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict


class fsdp(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.unique_params = self._get_unique_params()
        
        self.param_meta: Dict[torch.Tensor, dict] = {}
        self._shard_all_parameters()
        self._register_forward_hooks()

    def _get_unique_params(self):
        seen_ids = set()
        unique_params = []
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid not in seen_ids:
                seen_ids.add(pid)
                unique_params.append(p)
        return unique_params

    # 参数分片
    def _shard_all_parameters(self):
        for param in self.unique_params:
            meta = {}

            meta["is_sharded"] = True

            # 保存shape和dtype信息
            meta["full_shape"] = param.data.shape
            meta["dtype"] = param.data.dtype

            full_tensor = param.data

            dim0 = full_tensor.shape[0]
            shard_len = dim0 // self.world_size

            start = self.rank * shard_len
            end = start + shard_len

            local_shard = full_tensor[start:end].clone()

            meta["shard"] = local_shard
            meta["shard_len"] = shard_len

            self.param_meta[param] = meta

            # 清空完整parameter
            param.data = torch.zeros_like(param.data)

    # forward_pre_hook：
    # 前向执行之前触发（pre_fwd_hook）
    #
    # 把分片 gather，把完整权重写回 p.data，给计算用
    #
    # forward_hook：
    # 前向计算跑完之后触发
    #
    # 把完整权重清理掉，只保留本地分片
    def _pre_fwd_hook(self, module, inp):
        for p in module.parameters(recurse=False):
            if p not in self.param_meta:
                continue

            meta = self.param_meta[p]

            if not meta["is_sharded"]:
                continue

            full_w = self._all_gather_one_param(p)

            # compute_dtype只用于forward计算
            # 不改变master shard
            if self.compute_dtype is not None:
                full_w = full_w.to(self.compute_dtype)

            p.data.copy_(full_w)

    # 前向执行完成之后跑的钩子
    def _post_fwd_hook(self, module, inp, out):
        for p in module.parameters(recurse=False):
            if p not in self.param_meta:
                continue
            if isinstance(module, nn.Embedding):
                continue
            # 把刚刚gather出来的完整权重清理
            self._free_full_param(p)

    def _register_forward_hooks(self):
        for submod in self.module.modules():
            # 只要是Linear或者Embedding，就给它挂上钩子
            if isinstance(submod, (nn.Linear, nn.Embedding)):
                submod.register_forward_pre_hook(self._pre_fwd_hook)
                submod.register_forward_hook(self._post_fwd_hook)

    # 调用dist.all_gather收集所有 rank 的 shard，拼接成完整张量
    def _all_gather_one_param(self, param: torch.Tensor):
        meta = self.param_meta[param]

        local_shard = meta["shard"]

        gather_list = [
            torch.empty_like(local_shard)
            for _ in range(self.world_size)
        ]

        # all_gather：把所有rank的shard收集到gather_list
        dist.all_gather(
            gather_list,
            local_shard
        )

        full_weight = torch.cat(
            gather_list,
            dim=0
        )

        return full_weight

    # 前向计算完成之后调用，把param.data里那份临时拼出来的完整权重清理掉
    # 绝对不能碰 meta["shard"]！
    # meta["shard"] 是本rank持久保存的master权重分片，要保留
    def _free_full_param(self, param):
        meta = self.param_meta[param]
        param.data.copy_(meta["shard"])

    # fsdp_on_after_backward 会调用本函数
    # 在 loss.backward()全部结束、optimizer.step()之前执行
    # 此时每个param.grad里面存的是完整的全局梯度
    def reduce_scatter_gradients(self):
        for param in self.unique_params:
            meta = self.param_meta[param]

            if not meta["is_sharded"]:
                continue

            if param.grad is None:
                continue

            full_grad = param.grad

            if full_grad.is_sparse:
                full_grad = full_grad.to_dense()
            
            if isinstance(param, torch.Tensor) and param.grad.is_sparse:
                param.grad = param.grad.to_dense()

            shard_len = meta["shard_len"]

            # 输出：
            # 本rank得到的梯度shard，shape和权重shard完全对齐
            output_shard_grad = torch.empty(
                (shard_len, *full_grad.shape[1:]),
                device=full_grad.device,
                dtype=full_grad.dtype
            )

            # 将完整梯度按dim0切分成world_size份
            grad_chunks = list(
                torch.chunk(
                    full_grad,
                    self.world_size,
                    dim=0
                )
            )

            # reduce_scatter：
            # 多rank求和，再分发对应分片
            dist.reduce_scatter(output_shard_grad,grad_chunks)
            output_shard_grad.div_(self.world_size)
            meta["shard_grad"] = output_shard_grad

    # 把所有 GPU 上的碎片全部收集回来，拼成一份完整的模型 state_dict
    def gather_full_params(self) -> Dict[str, torch.Tensor]:
        out_dict = {}

        param_to_name = {
            p: name
            for name, p in self.module.named_parameters()
        }

        for param in self.unique_params:
            meta = self.param_meta[param]

            param_name = param_to_name[param]

            if meta["is_sharded"]:
                full_tensor = self._all_gather_one_param(param)
                full_tensor = full_tensor.to(torch.float32)
            else:
                full_tensor = param.data.clone()

            out_dict[param_name] = full_tensor.clone()

        return out_dict

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)



