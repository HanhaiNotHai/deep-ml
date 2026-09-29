import numpy as np

def calculate_contrast(img) -> int:
	"""
	Calculate the contrast of a grayscale image.
	Args:
		img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.
	"""
	min_val = 255
    max_val = 0
    for row in img:
        for x in row:
            min_val = min(x, min_val)
            max_val = max(x, max_val)
    return max_val - min_val