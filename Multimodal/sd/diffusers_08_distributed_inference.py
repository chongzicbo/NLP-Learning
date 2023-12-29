"""
如果出现version `GLIBCXX_3.4.29' not found ，
则：export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/sshadmin/bocheng/soft/installed/miniconda3/envs/torch/lib/
"""
import os

os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/sshadmin/bocheng/.cache/huggingface/hub/"
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

# 1.accelerate分布式推理

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
# distributed_state = PartialState()
# pipeline.to(distributed_state.device)
# with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
#     result = pipeline(prompt).images[0]
#     result.save(f"result_{distributed_state.process_index}.png")

# 2.pytorch分布式推理
import torch.distributed as dist
import torch.multiprocessing as mp


def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    pipeline.to(rank)
    if torch.distributed.get_rank() == 0:
        prompt = "adog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    image = pipeline(prompt).images[0]
    image.save(f"./{'_'.join(prompt)}.png")


def main():
    world_size = 2
    mp.spawn(run_inference, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
