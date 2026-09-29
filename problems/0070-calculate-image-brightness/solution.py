
def calculate_brightness(img):
	if not img:
		return -1
	
	len_row = len(img[0])
	s = 0
	for row in img:
		if len(row) != len_row:
			return -1
		for x in row:
			if x < 0 or x > 255:
				return -1
			s += x
	return s / (len(img) * len_row)
