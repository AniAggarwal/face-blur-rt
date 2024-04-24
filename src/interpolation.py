import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.patches as patches


box1 = [[10, 10, 20, 0]]  # Example keyframe box 1
box2 = [[50, 50, 70, 30]]  # Example keyframe box 2

x1 = np.array([box1[0][0], box2[0][0]])
y1 = np.array([box1[0][1], box2[0][1]])

x2 = np.array([box1[0][2], box2[0][2]])
y2 = np.array([box1[0][3], box2[0][3]])

# Create a cubic spline interpolator
t = [0, 1]  # Parameter values for the endpoints
cs_x1 = CubicSpline(t, x1)
cs_y1 = CubicSpline(t, y1)

cs_x2 = CubicSpline(t, x2)
cs_y2 = CubicSpline(t, y2)

# Evaluate the spline at non-linearly spaced points to mimic denser ends
num_points = 3
t_interp = (np.cos(np.linspace(0, np.pi, num_points)) + 1) / 2  # Map cosine to 0-1
x1_interp = cs_x1(t_interp)
y1_interp = cs_y1(t_interp)

x2_interp = cs_x2(t_interp)
y2_interp = cs_y2(t_interp)

# Plot the original points and the interpolated spline
plt.plot(box1[0][0], box1[0][1], "o", label="Keyframe 1")
plt.plot(box2[0][0], box2[0][1], "o", label="Keyframe 2")
plt.plot(x1_interp, y1_interp, "o-", label="Interpolated Motion")
plt.plot(x2_interp, y2_interp, "o-", label="Interpolated Motion")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Cubic Spline Interpolation with Density Variation")
plt.grid(True)
#plt.show()

interp_bboxes = np.array([item for item in zip(x1_interp, y1_interp, x2_interp, y2_interp)])
print(interp_bboxes[1])
print(interp_bboxes)

# Create figure and axes
fig, ax = plt.subplots()

# Plot each bounding box
for bbox in interp_bboxes:
    x1, y1, x2, y2 = bbox
    # Create a Rectangle patch
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)

# Set limits and aspect
ax.set_xlim(0, 80)  # Adjust according to your data range
ax.set_ylim(0, 60)  # Adjust according to your data range
ax.set_aspect('equal')  # Keep aspect ratio of the plot square

# Show the plot
plt.show()