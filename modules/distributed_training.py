import os
import torch
import torch.distributed as dist

# ==== Setup Process Group ====
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# ==== Cleanup Process Group ====
def cleanup():
    dist.destroy_process_group()
