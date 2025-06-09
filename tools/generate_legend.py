import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Data from color_legend.py
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

def create_legend(class_names, colors, filename='legend.png'):
    """
    Creates a legend image with color boxes and class names.
    """
    item_height = 30
    box_size = 20
    text_offset = 10
    left_padding = 10
    top_padding = 10
    
    # Use a basic font
    try:
        font = ImageFont.truetype("dejavusans.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Determine image dimensions
    max_text_width = 0
    for name in class_names:
        text_width, _ = font.getsize(name)
        if text_width > max_text_width:
            max_text_width = text_width
            
    img_width = left_padding + box_size + text_offset + max_text_width + left_padding
    img_height = top_padding * 2 + len(class_names) * item_height

    # Create image
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    # Draw items
    for i, (name, color_rgba) in enumerate(zip(class_names, colors)):
        y_pos = top_padding + i * item_height
        
        # Color box
        box_coords = [
            (left_padding, y_pos),
            (left_padding + box_size, y_pos + box_size)
        ]
        # Convert RGBA to RGB for filling the box
        color_rgb = tuple(color_rgba[:3])
        draw.rectangle(box_coords, fill=color_rgb)
        
        # Class name
        text_x = left_padding + box_size + text_offset
        text_y = y_pos + (box_size - font.getsize(name)[1]) // 2 # Center text vertically
        draw.text((text_x, text_y), name, fill='black', font=font)
        
    img.save(filename)
    print(f"Legend saved to {filename}")

if __name__ == '__main__':
    create_legend(class_names, colors, '../../misc/cgformer/legend.png') 