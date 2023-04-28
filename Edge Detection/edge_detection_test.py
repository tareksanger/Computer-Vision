import os
import numpy as np
import cv2
import time

from module.edge_detection import  edge_detection


# Sigma (Change here if you like)
sigma = 1

# Grab Image from Directory
img_dir = os.path.dirname('img/')
img_file_name = 'cat'  # Change Image Name HERE if testing with new Image
extension = '.jpg'      # Change Extension if necessary

# Read Image as Gray Scale
gray_scale_img = cv2.imread(os.path.join(img_dir, img_file_name + extension), flags=0)

# Run the Edge Detection and start a timer
t0 = time.perf_counter()
nms_img, mag_img, ori_img = edge_detection(gray_scale_img, sigma) 
t1 = time.perf_counter()
print ("Elapsed: {0:.3f}s".format(t1-t0))

# Save the Results to the 'result' Directory
cv2.imwrite(f"result/{img_file_name}_non_maximum_supperession{extension}", nms_img) # Non-Maximum Suppression before Sticks Filter
cv2.imwrite(f"result/{img_file_name}_gradient_magnitude{extension}", mag_img)       # Gradient Magnitude Image
cv2.imwrite(f"result/{img_file_name}_gradient_orientation{extension}", ori_img)     # Gradient Orientation Image

# Display the results
cv2.imshow("Original", gray_scale_img)
cv2.imshow('Non-Maximum Supperession', nms_img)
cv2.imshow('Gradient Magnitude', mag_img)
cv2.imshow('Gradient Orientation', ori_img)

cv2.waitKey(0)
cv2.destroyAllWindows() 

