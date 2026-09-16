import cv2
import numpy as np

# Load the three images
img1 = cv2.imread('blue_channel.png')
img2 = cv2.imread('green_channel.png')
img3 = cv2.imread('red_channel.png')

# Make sure all images are the same size (resize to match the first one if needed)
h, w = img1.shape[:2]
img2 = cv2.resize(img2, (w, h))
img3 = cv2.resize(img3, (w, h))

# --- Option A: Max intensity per-channel (works directly on BGR color images) ---
merged_color = np.maximum(np.maximum(img1, img2), img3)
cv2.imwrite('merged_max_color.png', merged_color)

# --- Option B: Max intensity based on grayscale brightness, but keep original color ---
# Useful when you want to pick the "brightest" full pixel from whichever image wins,
# rather than mixing channels independently.
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

stack_gray = np.stack([gray1, gray2, gray3], axis=0)   # shape: (3, H, W)
stack_color = np.stack([img1, img2, img3], axis=0)     # shape: (3, H, W, 3)

# Index of the image with the highest intensity at each pixel
winner_idx = np.argmax(stack_gray, axis=0)             # shape: (H, W)

# Pick the corresponding color pixel from the winning image
merged_by_brightness = np.take_along_axis(
    stack_color, winner_idx[None, :, :, None], axis=0
)[0]

cv2.imwrite('merged_max_brightness.png', merged_by_brightness)

print("Done — saved merged_max_color.png and merged_max_brightness.png")