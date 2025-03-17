import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_current_target().backend
if DEVICE == "cuda":
    DEVICE = "cuda:0"

print(DEVICE)


@triton.jit
def subtraction_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x - y
    tl.store(output_ptr + offsets, output, mask=mask)


def subtraction(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    print(output.device)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    subtraction_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x - y
output_triton = subtraction(x, y)
print(
    f"The maximum difference between torch and triton is "
    f"{torch.max(torch.abs(output_torch - output_triton))}"
)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different values for `size`
        x_log=True,  # x-axis is logarithmic
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton",
            "torch",
        ],  # Argument names to use as a line in the plotwhose value corresponds to a different line in the plot
        line_names=["Triton", "Torch"],  # Name of the lines
        styles=[("blue", "-"), ("green", "-")],  # Line styles
        ylabel="GB/s",  # Label for the y-axis
        plot_name="vector_subtraction_performance",  # Name of the plot
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x - y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: subtraction(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)


benchmark.run(print_data=True, show_plots=True, save_path="./output")
