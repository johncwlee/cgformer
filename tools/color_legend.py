import numpy as np

class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
    'vegetation', 'terrain', 'pole', 'traffic-sign', 'other-structure', 'other-object'
]
colors = np.array(
	[
		[100, 150, 245, 255],
		[100, 230, 245, 255],
		[30, 60, 150, 255],
		[80, 30, 180, 255],
		[100, 80, 250, 255],
		[255, 30, 30, 255],
		[150, 30, 90, 255],
		[255, 0, 255, 255],
		[255, 150, 255, 255],
		[75, 0, 75, 255],
		[175, 0, 75, 255],
		[255, 200, 0, 255],
		[255, 120, 50, 255],
		[0, 175, 0, 255],
		[135, 60, 0, 255],
		[150, 240, 80, 255],
		[255, 240, 150, 255],
		[255, 0, 0, 255],
		[0, 0, 0, 255]
	]).astype(np.uint8)