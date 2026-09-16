import cv2

# Load the image
img = cv2.imread('merged_max_brightness.png')

# Convert to grayscale first (binary conversion needs a single channel)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Stretch contrast — helps a lot when the source image is low-contrast/faint
norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

# Simple fixed threshold (pixels > 127 -> white, else black)
_, binary = cv2.threshold(norm, 127, 255, cv2.THRESH_BINARY)

# Otsu's method — automatically picks the best threshold value
# (usually better than a fixed value for varying image conditions)
_, binary_otsu = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite('binary_simple.png', binary)
cv2.imwrite('binary_otsu.png', binary_otsu)