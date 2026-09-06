import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List


class SimpleDDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.pending_works: List[dist.Work] = []

        # 先广播初始化参数
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        self._register_hooks()

    def _register_hooks(self):
        registered_ids = set()

        def post_grad_hook(param: torch.nn.Parameter):
            # 梯度已经完全累加完毕，param.grad是最终梯度
            if self.world_size == 1:
                return
            # 直接对param.grad发起异步all_reduce
            work = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True
            )
            self.pending_works.append(work)

        for param in self.module.parameters():
            if param.requires_grad and id(param) not in registered_ids:
                param.register_post_accumulate_grad_hook(post_grad_hook)
                registered_ids.add(id(param))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def wait_all_communications(self):
        if self.world_size == 1:
            self.pending_works.clear()
            return

        # 等待所有异步梯度通信完成
        for work in self.pending_works:
            work.wait()

        # 求和梯度 → 全局平均梯度
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad /= self.world_size

        self.pending_works.clear()
