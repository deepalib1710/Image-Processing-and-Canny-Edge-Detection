# Image Processing and Canny Edge Detection

A Python-based image processing project that demonstrates various computer vision techniques including filtering, edge detection, and morphological operations using OpenCV.

## Overview

This project implements a comprehensive image processing pipeline that showcases fundamental computer vision algorithms:
- **Image Filtering**: Box filtering and bilateral filtering for noise reduction
- **Image Enhancement**: Sharpening techniques
- **Edge Detection**: Canny edge detection algorithm
- **Morphological Operations**: Erosion, dilation, opening, closing, gradients, and top/black hat transforms

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Installation

```bash
pip install opencv-python numpy matplotlib
```

## Algorithm Pipeline

### Step 1: Image Loading & Preprocessing
- Load image from file path
- Resize image to 500x500 pixels for standardized processing
- Display original and resized images

### Step 2: Noise Reduction & Filtering
- **Box Filter**: Apply averaging filter with 80×80 kernel to smooth the image
- **Bilateral Filter**: Apply bilateral filtering (d=9, sigmaColor=75, sigmaSpace=75) to preserve edges while reducing noise

### Step 3: Image Enhancement
- **Sharpening**: Apply weighted combination to enhance edges using the formula: `enhanced = 1.5 × original - 0.5 × blurred`

### Step 4: Edge Detection
- **Canny Edge Detection**: 
  - Apply Canny algorithm with lower threshold (100) and upper threshold (200)
  - Use 3×3 aperture size with L2 gradient
  - Detect sharp intensity gradients and edges in the image

### Step 5: Binary Image Processing
- Apply binary thresholding with threshold value of 127
- Invert binary image to prepare for morphological operations
- Create 5×5 structuring element kernel

### Step 6: Morphological Operations
Apply 6 different morphological transformations on the binary image:

1. **Erosion**: Shrink white regions by applying minimum filter with 1 iteration
2. **Dilation**: Expand white regions by applying maximum filter with 1 iteration
3. **Opening**: Erosion followed by dilation - removes small objects and noise
4. **Closing**: Dilation followed by erosion - fills small holes in objects
5. **Morphological Gradient**: Difference between dilation and erosion - highlights object boundaries
6. **Top Hat**: Difference between original and opening - extracts small elements
7. **Black Hat**: Difference between closing and original - extracts small holes
8. **Binary Image**: Reference binary image for comparison

### Step 7: Visualization
- Display all 8 results in a 2×4 grid layout using Matplotlib
- Show each result with appropriate grayscale colormap and title

## Key Concepts

**Filtering**: Reduces noise and smooths images using convolution operations

**Edge Detection**: Identifies boundaries and sharp transitions in pixel intensity

**Morphological Operations**: Modifies image structure using set operations with a structuring element

**Binary Image Processing**: Converts grayscale to binary (black and white) for structural analysis

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Canny Edge Detection](https://en.wikipedia.org/wiki/Canny_edge_detector)
- [Morphological Operations](https://en.wikipedia.org/wiki/Mathematical_morphology)
