import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Iterable, Type


class Sharded_Optimizer(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], optimizer_cls: Type[Optimizer], **kwargs):
        super().__init__(params, defaults=kwargs)
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.global_params = list(self.param_groups[0]["params"])
        self.unique_params = self._get_unique_params(self.global_params)
        self.local_params = self._sharded_params(self.unique_params) #分片参数
        self.local_optimizer = self.optimizer_cls(self.local_params, **kwargs)

    def _get_unique_params(self, global_params: list[torch.Tensor]) -> list[torch.Tensor]:
        seen_ids = set()
        unique_params = []
        for p in global_params:
            tensor_id = id(p)
            if tensor_id not in seen_ids and p.requires_grad:
                seen_ids.add(tensor_id)
                unique_params.append(p)
        return unique_params

    #不是切分张量 只是选出一份参数子集 优化器只更新
    def _sharded_params(self, global_params: list[torch.Tensor]):

        #id() 是 Python 内置函数：返回对象在内存中的唯一整数标识（内存地址编号）。
        #set()指的是建立集合，实现去重。防止同一个参数在模型不同地方出现，被不同优化器同时优化
        seen_ids = set()
        unique_params = []

        for p in global_params:
            tensor_id = id(p)
            if tensor_id not in seen_ids:
                seen_ids.add(tensor_id)
                unique_params.append(p)
        local_params = []
        for idx, params in enumerate(unique_params):
            if idx % self.world_size == self.rank:
                local_params.append(params)
        return local_params
    
    def _broadcast_updated_params(self):
        for idx, param in enumerate(self.unique_params):
            # 算出这个参数归哪个rank负责更新
            owner_rank = idx % self.world_size
            # 全部rank都执行这一条broadcast；owner_rank发送，其余rank接收
            dist.broadcast(param.data, src=owner_rank, async_op=False)
            #src：发送源进程编号.async_op=False：同步阻塞调用。函数返回前，通信一定完成；执行完这一行，param.data 的内容已经是广播完成之后的数据。

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.local_optimizer.step()
        self._broadcast_updated_params()
        return loss

    def zero_grad(self, set_to_none: bool = False):
        for p in self.global_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

