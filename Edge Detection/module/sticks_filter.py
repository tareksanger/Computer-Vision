'''
    Auther: Tarek Sanger
    Student Number: 101059686
    
    Course Code: COMP4102A
    Assigmenet 1
'''
import numpy as np
import cv2


def sticks_filter(img):
    n, m = img.shape

    # Prep the outout
    output = np.zeros(img.shape, dtype=np.uint8)
    
    #  Stick Kernel Values
    s = [
        cv2.filter2D(src=img, ddepth=-1, kernel=kernel_1()),
        cv2.filter2D(src=img, ddepth=-1, kernel=kernel_2()),
        cv2.filter2D(src=img, ddepth=-1, kernel=kernel_3()),
        cv2.filter2D(src=img, ddepth=-1, kernel=kernel_4()),
        cv2.filter2D(src=img, ddepth=-1, kernel=kernel_5()),
        cv2.filter2D(src=img, ddepth=-1, kernel=kernel_6()),
        cv2.filter2D(src=img, ddepth=-1, kernel=kernel_7()),
        cv2.filter2D(src=img, ddepth=-1, kernel=kernel_8()),
    ]

    for y in range(2,n-2):
        for x in range(2,m-2):
            # Parse the sub matrix from the image
            y_min, y_max = max(y-2, 0), min(y+3, n)
            x_min, x_max = max(x-2, 0), min(x+3, m)
            
            # Create T map
            i_hat = np.sum(img[y_min:y_max, x_min:x_max])/(5**2)
            t = {i: (s[i-1][y,x] - i_hat) for i in range(1, 9)}
            
            # Find the key for with the maximum value
            chosen_filter = max(t, key=t.get) - 1
            output[y,x] = s[chosen_filter][y,x] # Set the pixel with the corresponding kernel output  

    return output


''''
    Each of the 8 Kernels Hard Coded.
'''
def kernel_1():
    kernel_1 = np.zeros((5,5))
    for i in range(5):
        kernel_1[2, i] = 1/5
    return kernel_1

def kernel_2():
    kernel = np.zeros((5,5))
    kernel[3, 0] = 1/5
    kernel[3, 1] = 1/5
    kernel[2, 2] = 1/5
    kernel[1, 3] = 1/5
    kernel[1, 4] = 1/5
    return kernel

def kernel_3():
    kernel =  np.flipud(1/5 * np.identity(5))
    return kernel

def kernel_4():
    kernel =  np.flipud(1/5 * np.identity(5))
    kernel[0, 4] = 0
    kernel[0, 3] = 1/5
    kernel[4, 0] = 0
    kernel[4, 1] = 1/5
    return kernel

def kernel_5():
    kernel = np.zeros((5,5))
    for i in range(5):
        kernel[i, 2] = 1/5
    return kernel

def kernel_6():
    kernel = np.zeros((5,5))
    kernel[0, 1] = 1/5
    kernel[1, 1] = 1/5
    kernel[2, 2] = 1/5
    kernel[3, 3] = 1/5
    kernel[4, 3] = 1/5
    return kernel

def kernel_7():
    return 1/5 * np.identity(5)

def kernel_8():
    kernel = 1/5 * np.identity(5)
    kernel[0,0] = 0
    kernel[1, 0] = 1/5
    kernel[4,4] = 0
    kernel[1, 0] = 1/5
    return kernel
