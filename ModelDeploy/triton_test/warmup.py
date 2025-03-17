import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y):
    output = torch.zeros_like(x)  # 初始化为零
    n_elements = x.shape[0]
    BLOCK_SIZE = 256
    num_warps = 4
    grid = (triton.cdiv(n_elements, BLOCK_SIZE), 1, 1)

    # 预热核函数
    kernel = add_kernel.warmup(
        x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, grid=grid
    )
    kernel._init_handles()

    # 实际调用
    add_kernel[grid](
        x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024, device="cuda").contiguous()
    y = torch.randn(1024, device="cuda").contiguous()
    output = add(x, y)

    expected = x + y
    print("Triton output:", output[:5])
    print("Expected output:", expected[:5])

    assert torch.allclose(
        output, expected, rtol=1e-5, atol=1e-5
    ), "Results don't match!"
    print("Results match between Triton and PyTorch!")
