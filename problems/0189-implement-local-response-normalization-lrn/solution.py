import numpy as np

def local_response_normalization(x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W)
        n: Local window size parameter (half-width h = floor(n/2))
        k: Additive constant
        alpha: Scaling parameter
        beta: Exponent parameter

    Returns:
        Normalized tensor of same shape as input, rounded to 4 decimals.
    """
    h = n // 2
    window_size = 2 * h + 1  # 对称窗口的实际长度

    # 1. 计算激活值的平方
    x_sq = np.square(x)

    # 2. 通道维度前后补零，实现边缘自动裁剪的效果
    padded = np.pad(x_sq, pad_width=((0, 0), (h, h), (0, 0), (0, 0)), mode="constant")

    # 3. 通道维度滑动窗口求和（向量化，无循环）
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size, axis=1)
    sum_sq = windows.sum(axis=-1)

    # 4. 按公式计算归一化结果
    denominator = (k + alpha * sum_sq) ** beta
    output = x / denominator

    # 5. 保留4位小数返回
    return np.round(output, 4)
