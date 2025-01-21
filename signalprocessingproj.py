
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread(r'C:\Users\Krishna Kumar Banka\Desktop\flower.jpeg')

# Check if the image is loaded properly
if img is None:
    print("Error loading image.")
    exit()

# Display original image
cv2.imshow("Original Image", img)

# Resize the image
resized_img = cv2.resize(img, (500, 500))
cv2.imshow("Resized Image", resized_img)

# Box filtering (blurring)
output_box = cv2.boxFilter(resized_img, -1, (80, 80), normalize=True)
cv2.imshow('Box Filtered Image', output_box)

# Bilateral filtering for denoising
output_bil = cv2.bilateralFilter(resized_img, d=9, sigmaColor=75, sigmaSpace=75)
cv2.imshow('Bilateral Filtered Image', output_bil)

# Sharpen the image
sharpened = cv2.addWeighted(resized_img, 1.5, output_box, -0.5, 0)
cv2.imshow('Sharpened Image', sharpened)

# Canny edge detection
t_lower = 100  # Lower threshold
t_upper = 200  # Upper threshold
aperture_size = 3  # Aperture size (must be odd)
L2Gradient = True  # Use L2 gradient
edges = cv2.Canny(resized_img, t_lower, t_upper, apertureSize=aperture_size, L2gradient=L2Gradient)
cv2.imshow('Canny Edge Detection', edges)

# Apply binary thresholding
_, binary_img = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY_INV)

# Define a kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Erosion
erosion = cv2.erode(binary_img, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(binary_img, kernel, iterations=1)

# Opening (Erosion followed by Dilation)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

# Closing (Dilation followed by Erosion)
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

# Morphological Gradient (Difference between Dilation and Erosion)
gradient = cv2.morphologyEx(binary_img, cv2.MORPH_GRADIENT, kernel)

# Top Hat (Difference between input image and Opening)
top_hat = cv2.morphologyEx(binary_img, cv2.MORPH_TOPHAT, kernel)

# Black Hat (Difference between Closing and input image)
black_hat = cv2.morphologyEx(binary_img, cv2.MORPH_BLACKHAT, kernel)

# Display the morphological operation results using Matplotlib
plt.figure(figsize=(15, 10))
plt.subplot(2, 4, 1), plt.imshow(binary_img, cmap='gray'), plt.title('Binary Image')
plt.subplot(2, 4, 2), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
plt.subplot(2, 4, 3), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
plt.subplot(2, 4, 4), plt.imshow(opening, cmap='gray'), plt.title('Opening')
plt.subplot(2, 4, 5), plt.imshow(closing, cmap='gray'), plt.title('Closing')
plt.subplot(2, 4, 6), plt.imshow(gradient, cmap='gray'), plt.title('Gradient')
plt.subplot(2, 4, 7), plt.imshow(top_hat, cmap='gray'), plt.title('Top Hat')
plt.subplot(2, 4, 8), plt.imshow(black_hat, cmap='gray'), plt.title('Black Hat')

plt.tight_layout()
plt.show()

# Wait for a key press and close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()

