## Testing & Development Environment

Python version 3.7  
Packages:

- Numpy 1.21.2
- OpenCV (opencv-python) 4.5.4.60

### Notes:

When Testing Parts 2 and 3 the images will default to the cat2.jpg. When Running the tests they will create saves of each image, but will not overwrite the images I have submitted.

After Testing closing the images should terminate the program - I have noticed that at times the program will not terminate when all images have closed. Starting a new terminal should fix this.

python test_2.py

python test_3.py

## Edge Detection

The Algorithms for this part can be found in module/edge_detection.py.

To test the code run the edge_detection function, or simply run the edge_detection_test.py file. The edge_detection function returns 3 values, the Non-Maximum Suppression, The Gradient Magnitude and the Gradient Orientation results.

For the guassian_blur, the blur uses the separable filter methods when the value of Sigma is larger than 40. I did this for testing - For smaller values of sigma and smaller images simply multiplying the matrixes runs faster. The Code for this is found in the convolve_gaussian function. The Kernel is calculated in the gaussian_kernel1d function.

For the Sobel the the kernels for x and y are hard coded - They can be found in the sobel_edge function. The calculations for the gradient magnitude and gradient orientation can be found in the convolve_sobel. Sobel Edge returns the both gradient magnitude and gradient orientation.

The Non-Maximum Suppression can be found in the non_maximum_suppression. In this function I look at the orientation and round it to the nearest 45deg, this is used to then decide which pixel neighbors to look at.

## Sticks Filter

The Code for for Sticks Filter can be found in module/sticks_filter.py.

To test the code run the code run sticks_filter_test.py.
