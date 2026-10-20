from typing import Literal

import numpy as np


def flip_image(
    image: list[list[int]] | list[list[list[int]]], direction: Literal['horizontal', 'vertical']
):
    """
    Flip an image horizontally or vertically.

    Args:
        image: 2D or 3D list/array representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'

    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """

    if direction != 'horizontal' and direction != 'vertical':
        return -1

    def T(image: list[list[int]] | list[list[list[int]]]):
        return list(zip(*image))

    if direction == 'horizontal':
        image = T(image)

    image = image[::-1]

    if direction == 'horizontal':
        image = T(image)

    return list(map(list, image))
