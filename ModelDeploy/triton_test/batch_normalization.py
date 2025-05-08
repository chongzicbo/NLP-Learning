import triton
import triton.language as tl
import torch


@triton.jit
def batch_norm_forward_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    gamma_ptr,  # 缩放参数指针
    beta_ptr,  # 平移参数指针
    mean_ptr,  # 均值指针
    var_ptr,  # 方差指针
    n_cols,  # 每个样本的特征维度
    eps,  # 数值稳定性常数
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)  # 当前处理的样本索引
    col_offsets = tl.arange(0, BLOCK_SIZE)  # 列偏移量
    mask = col_offsets < n_cols  # 防止越界

    # 加载输入数据
    x_ptrs = x_ptr + row_idx * n_cols + col_offsets
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # 计算均值
    mean = tl.sum(x, axis=0) / n_cols

    # 计算方差 - 使用乘法而不是幂运算
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols

    # 归一化
    x_hat = (x - mean) / tl.sqrt(var + eps)

    # 加载 gamma 和 beta
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)

    # 缩放和平移
    y = gamma * x_hat + beta

    # 存储结果
    y_ptrs = y_ptr + row_idx * n_cols + col_offsets
    tl.store(y_ptrs, y, mask=mask)

    # 存储均值和方差（可选）
    if row_idx == 0:
        tl.store(mean_ptr + col_offsets, mean, mask=mask)
        tl.store(var_ptr + col_offsets, var, mask=mask)


def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    使用 Triton 实现 Batch Normalization 前向传播。
    :param x: 输入张量 (batch_size, num_features)
    :param gamma: 缩放参数 (num_features,)
    :param beta: 平移参数 (num_features,)
    :param eps: 数值稳定性常数
    :return: 归一化后的张量
    """
    batch_size, num_features = x.shape

    # 初始化输出张量
    y = torch.empty_like(x)
    mean = torch.empty(num_features, device=x.device)
    var = torch.empty(num_features, device=x.device)

    # 定义块大小
    BLOCK_SIZE = 1024

    # 启动 Triton 内核
    grid = lambda meta: (batch_size,)
    batch_norm_forward_kernel[grid](
        x, y, gamma, beta, mean, var, num_features, eps, BLOCK_SIZE
    )

    return y, mean, var


# 测试代码
if __name__ == "__main__":
    # 输入张量
    batch_size = 32
    num_features = 1024
    x = torch.randn(batch_size, num_features, device="cuda")
    gamma = torch.ones(num_features, device="cuda")
    beta = torch.zeros(num_features, device="cuda")

    # 执行 Batch Normalization
    y, mean, var = batch_norm_forward(x, gamma, beta)

    print("输出张量形状:", y.shape)
    print("均值:", mean)
    print("方差:", var)
