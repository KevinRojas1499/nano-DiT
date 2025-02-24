import torch
import torch.distributed as dist

# Wrapper for torch distributed

class ReduceOp:
    SUM = dist.ReduceOp.SUM
    PRODUCT = dist.ReduceOp.PRODUCT
    MIN = dist.ReduceOp.MIN
    MAX = dist.ReduceOp.MAX
    BAND = dist.ReduceOp.BAND
    BOR = dist.ReduceOp.BOR
    BXOR = dist.ReduceOp.BXOR
    AVG = dist.ReduceOp.AVG

def initialize_dist(global_seed):
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = global_seed * rank + world_size
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    return device, rank, world_size, seed

def reduce(tensor, reduce_op):
    return dist.all_reduce(tensor, reduce_op)

def destroy():
    dist.destroy_process_group()

def gather_object(output_list, object):
    dist.all_gather_object(output_list, object)