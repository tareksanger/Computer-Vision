'''
    Auther: Tarek Sanger
    Student Number: 101059686
    
    Course Code: COMP4102A
    Assigmenet 1
'''
import math
import numpy as np

def edge_detection(img, sigma: int):

    # Step 1: Smooth the Image to reduce noise using Gaussian Blur
    smoothed_img = gaussian_blur(img, sigma=sigma)

    # Step 2: Detect Edges
    magnitude_img, orientation_img = sobel_edge(smoothed_img)

    # Step 2: Enhance Edges
    non_maximum_suppression_img = non_maximum_suppression(magnitude_img, orientation_img)

    return non_maximum_suppression_img, magnitude_img, orientation_img

def sobel_edge(img, threshold = 25):

    '''
        Sobel Edge Detection.

        Img: an n*m Matrix of pixels

        Returns the Magnitude Strenth image and the Gradient Direction 
    '''
    # The Sobel Filters
    kernel_x = np.matrix([[1,0,-1], [2, 0, -2],[1,0,-1]])
    kernel_y = kernel_x.T

    # Convole using sobel
    mag_img, ori_img = convolve_sobel(img, kernel_x, kernel_y, threshold)

    return mag_img, ori_img

def convolve_sobel(img, kernel_x, kernel_y, threshold=15):

    # height and width of the image
    h, w = img.shape

    # Prep the output
    mag_img = np.zeros(img.shape, dtype=np.uint8)
    ori_img = np.zeros(img.shape, dtype=np.float32)


    # Grab the height and width of the kernel
    k_h = kernel_x.shape[0]
    k_h_half = k_h//2 
    k_w = kernel_x.shape[1]
    k_w_half = k_w//2

    for y in range(h):
        for x in range(w):

            # Grab the available range of pixels Range
            y_min, y_max = max(y-k_h_half, 0), min(y+k_h_half+1, h)
            x_min, x_max = max(x-k_w_half, 0), min(x+k_w_half+1, w)
            

            # Parse the kernel and the image pixels based on the x, y, min and max values
            # This is to handle the boarder pixels. Here we are simply ignoring the missing pixels all together.
            kern_x = kernel_x[y_min-y + k_h_half:k_h - y - k_h_half + y_max - 1, x_min-x + k_w_half: k_w - x - k_w_half + x_max - 1] 
            kern_y = kernel_y[y_min-y + k_h_half:k_h - y - k_h_half + y_max - 1, x_min-x + k_w_half: k_w - x - k_w_half + x_max - 1] 
            snap = img[y_min:y_max, x_min:x_max]
            
            # Calculate I_x and I_y
            sobel_x = np.sum(np.multiply(snap, kern_x), dtype=np.float32)
            sobel_y = np.sum(np.multiply(snap, kern_y), dtype=np.float32)

            # Calculate the Gradient Magnitude  and apply the threshold
            gradient_magnitude = np.hypot(sobel_x, sobel_y)
            
            
            mag_img[y, x] = gradient_magnitude if gradient_magnitude >= threshold else 0

            # Calculate the orientation
            orientation = np.arctan2(sobel_y, sobel_x)
            ori_img[y, x] = np.rad2deg(orientation)

    return mag_img, ori_img

def gaussian_blur(img, sigma: int):
    '''
        
    '''
    # Prepare the output.
    output = np.zeros(img.shape, dtype=img.dtype.name)

    # Calcualte the size of the kernel and the kernel
    hsize = 2 * math.ceil(3 * sigma) + 1 
    kernel = gaussian_kernel1d(size=hsize, sigma=sigma)
    
    if sigma >= 40: # For Sigma values greater than 40 use the seperable matrix technique.
        gausssian_img = convolve_gaussian(img, kernel)
        convolve_gaussian(gausssian_img, kernel.T, output)
    else: 
        # Use simple Matrix Multiplication
        kernel = np.matmul(kernel.T, kernel)
        convolve_gaussian(img, kernel, output)
    return output

def convolve_gaussian(img, kernel, output = None):

    # height and width of the image
    h, w = img.shape

    # Create the output if none was defined
    if output is None:
        output = np.zeros(img.shape, dtype=img.dtype.name)

    # Grab the height and width of the kernel
    k_h = kernel.shape[0]
    k_h_half = k_h//2 
    k_w = kernel.shape[1]
    k_w_half = k_w//2

    for y in range(h):
        for x in range(w):

            # Grab the available range of pixels Range
            y_min, y_max = max(y-k_h_half, 0), min(y+k_h_half+1, h)
            x_min, x_max = max(x-k_w_half, 0), min(x+k_w_half+1, w)

            # Parse the kernel and the image pixels based on the x, y, min and max values
            # This is to handle the boarder pixels. Here we are simply ignoring the missing pixels all together.
            kern = kernel[y_min-y + k_h_half:k_h - y - k_h_half + y_max - 1, x_min-x + k_w_half: k_w - x - k_w_half + x_max - 1] # pa
            snap = img[y_min:y_max, x_min:x_max]
            # Calculate the blured pixel
            output[y, x] = np.sum(np.multiply(kern, snap)) / np.sum(kern)
    return output

def gaussian_kernel1d(size: int = 7, sigma:int = 1):
    """Returns a 1D Gaussian kernel."""
    # Function used to determine the value of each pixel in the kernel
    kernel = lambda h, k: (1/(2 * math.pi * sigma ** 2)) ** math.e ** (-(((h**2) + (k**2))/(2*sigma**2)))
    
    # Create the 1d kernal
    r = range(-int(size//2),int(size//2 + 1))
    kernel1d = np.matrix([kernel(x, 3) for x in r]) # Build an the Edge of the 
    return kernel1d

def non_maximum_suppression(img, orientation_matrix):
    n, m = img.shape

    # Prep the output
    output = np.zeros(img.shape, dtype=img.dtype.name)


    for j in range(1,n-1):
        for i in range(1,m-1):
            
            #  Round to the nearest 45 deg
            angle = round(orientation_matrix[j,i]/45) * 45

            if angle in [-180, 0, 180]:     # Horizontal 0
                a = img[j, i+1]
                b = img[j, i-1]
            elif angle in [-135, 45, 225]:  # Diagonal 45
                a = img[j+1, i+1]
                b = img[j-1, i-1]
            elif angle in [-90, 90, 270]:   # Vertical 90
                a = img[j+1, i]
                b = img[j-1, i]
            elif angle in [-45, 135, 315]:  # Diagonal 135
                a = img[j+1, i-1]
                b = img[j-1, i+1]

            # Non-max Suppression
            output[j,i] = img[j,i] if (img[j,i] >= a) and (img[j,i] >= b) else 0

    return output



