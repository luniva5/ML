import cv2
import numpy as np

# Provide the full absolute path to the image file
image_path = 'rectangle.png'

# Load the image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using the Canny edge detector
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a list to store rectangle information (x, y, w, h, line_size)
rectangle_info = []

# Loop through the detected contours
for contour in contours:
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Filter out very small rectangles
    if w > 10 and h > 10:
         # Find edges within the bounding rectangle
        roi_edges = edges[y:y+h, x:x+w]
        
        # Calculate the length of the detected edges within the rectangle
        line_length = np.sum(roi_edges) / 255
        
        # Store the rectangle information
        rectangle_info.append((x, y, w, h, line_length))

# Sort the rectangle information based on contour lengths
rectangle_info.sort(key=lambda info: info[4])

# Assign unique numbers to rectangles based on the sorted order
assigned_number = 1
for x, y, w, h, _ in rectangle_info:
    cv2.putText(image, str(assigned_number), (x + w // 2, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    assigned_number += 1

# Display the image with numbering
cv2.imshow('Numbered Rectangles Based on Line Length', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
