import os
import numpy as np
import cv2
import time

from module.sticks_filter import sticks_filter
from module.edge_detection import  gaussian_blur, non_maximum_suppression, sobel_edge

# Sigma (Change here if you like)
sigma = 1

# Grab Image from Directory
img_dir = os.path.dirname('img/')
img_file_name = 'cat'  # Change Image Name HERE if testing with new Image
extension = '.jpg'      # Change Extension if necessary

# Read Image as Gray Scale
gray_scale_img = cv2.imread(os.path.join(img_dir, img_file_name + extension), flags=0)

# Run the Edge Detection With Sticks Filter and start a timer
t0 = time.perf_counter()
gauss_img = gaussian_blur(img=gray_scale_img, sigma=sigma)        # Gaussian Blur
mag_img, ori_img = sobel_edge(img=gauss_img)                # Sobel Edge
sticks_img = sticks_filter(img=mag_img)                     # Sticks Filter 
nms_img = non_maximum_suppression(sticks_img, ori_img)  # Non-Maximum Suppression
t1 = time.perf_counter()
print ("Elapsed: {0:.3f}s".format(t1-t0))

# Save the Results to the 'Edge detection' Directory
cv2.imwrite(f"Edge detection/{img_file_name}_sticks_filter_on_gradient_magnitude{extension}", sticks_img)
cv2.imwrite(f"Edge detection/{img_file_name}_non_maximum_supperession_WITH_sticks_filter{extension}", nms_img)

# Display the results
cv2.imshow("Original", gray_scale_img)
cv2.imshow('Gradient Magnitude', mag_img)
cv2.imshow('Sticks Filter', sticks_img)
cv2.imshow('Non-Maximum Supperession WITH Sticks Filter', nms_img)

cv2.waitKey(0)
cv2.destroyAllWindows() 


