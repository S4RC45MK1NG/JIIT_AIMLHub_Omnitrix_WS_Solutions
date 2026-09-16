import cv2
import numpy as np

# Load the image
img = cv2.imread('oho.png')

# OpenCV loads images in BGR order (not RGB!)
b, g, r = cv2.split(img)

# If you want true RGB order instead of BGR:
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(img_rgb)

# Save each channel as a grayscale image
cv2.imwrite('blue_channel.png', b)
cv2.imwrite('green_channel.png', g)
cv2.imwrite('red_channel.png', r)

# Optional: visualize each channel in its own color (instead of grayscale)
zeros = np.zeros_like(b)
blue_colored  = cv2.merge([b, zeros, zeros])
green_colored = cv2.merge([zeros, g, zeros])
red_colored   = cv2.merge([zeros, zeros, r])

cv2.imwrite('blue_colored.png', blue_colored)
cv2.imwrite('green_colored.png', green_colored)
cv2.imwrite('red_colored.png', red_colored)
