import numpy as np


def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    input_matrix = np.pad(input_matrix, padding)
    output_matrix = np.empty((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output_matrix[i, j] = (
                input_matrix[
                    i * stride : i * stride + kernel_height,
                    j * stride : j * stride + kernel_width,
                ]
                * kernel
            ).sum()

    return output_matrix
