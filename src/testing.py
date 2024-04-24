import torch
import utils
import numpy as np
from scipy.interpolate import CubicSpline
""" print(f'PyTorch version: {torch.__version__}')
print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')
 """
'''
Ultra-Light-Fast-Generic-Face-Detector-1MB (ULFG): This model stands out due to its extremely small size and high speed, designed specifically for edge devices. The model achieves this efficiency by using a very streamlined architecture that reduces computational requirements. While it might not match YUNet's accuracy, its inference time is remarkably low.

BlazeFace: Developed by Google, BlazeFace is designed for mobile real-time face detection. It uses a lightweight convolutional neural network inspired by the MobileNet architecture, optimized for speed and low-power mobile applications. It's specifically tailored for frontal face detection and might offer competitive speed, particularly on mobile devices.
'''

# Given original bounding boxes
original_boxes = [
    [[10, 20, 30, 40]],  # Example keyframe box 1
    [[50, 60, 70, 80]],  # Example keyframe box 2
    [[90, 100, 110, 120]]  # Example keyframe box 3
]

# Number of interpolated boxes between each keyframe
num_interpolated_boxes = 10

# Create an array of t values (0 to 1) for interpolation
t_values = np.linspace(0, 1, num_interpolated_boxes + 2)[1:-1]  # Exclude endpoints

# Separate x and y coordinates for each corner of the bounding boxes
x_coords = np.array([[box[0][0], box[0][2]] for box in original_boxes])
y_coords = np.array([[box[0][1], box[0][3]] for box in original_boxes])

# Perform cubic spline interpolation for x and y coordinates
cs_x = CubicSpline(range(len(original_boxes)), x_coords, axis=0)
cs_y = CubicSpline(range(len(original_boxes)), y_coords, axis=0)

# Generate interpolated bounding boxes
interpolated_boxes = []
for t in t_values:
    interpolated_x = cs_x(t)
    interpolated_y = cs_y(t)
    interpolated_boxes.append([[interpolated_x[0], interpolated_y[0], interpolated_x[1], interpolated_y[1]]])

# Combine original and interpolated boxes
all_boxes = original_boxes + interpolated_boxes

# Print the resulting list of bounding boxes
for i, box in enumerate(all_boxes):
    print(f"Box {i + 1}: {box}")
