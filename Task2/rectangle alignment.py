import cv2
import numpy as np

# Load the image
image = cv2.imread('rectangle.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using the Canny edge detector
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the detected contours
for contour in contours:
    # Filter out very small contours
    if cv2.contourArea(contour) > 100:
        # Get the rotated bounding box of the rectangle
        rect = cv2.minAreaRect(contour)
        angle = rect[-1]

        # Rotate the individual rectangle to align it
        (h, w) = image.shape[:2]
        center = rect[0]
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        aligned_rect = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Display the aligned rectangle
        cv2.imshow('Aligned Rectangle', aligned_rect)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
