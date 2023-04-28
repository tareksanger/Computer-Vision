import os
import cv2 
import numpy as np
import scipy.ndimage.filters as filters


def main():

    quality_lvl = 2
    max_quality_lvl = 4
    title = 'Harris Corner Detection'

    img_dir = os.path.abspath('img/')
    img = os.path.join(img_dir, 'box_in_scene.png')

    src = cv2.imread(cv2.samples.findFile(img))

    # Step 1 ------------------------
    cv2.imshow('Box in Scene', src)


    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)



    # Step 2 ------------------------
    min_eigen_val = cv2.cornerMinEigenVal(src_gray, 3, 3) # Calculate Min Eigen Values


    # Step 3 & 4 ------------------------
    # Step 3 & 4 where combined in this function so that the images could both update with the track bar
    def on_trackbar(val):
        ''' 
            Used to display the track bar and track bar results.

            val is a number between 0 and 4, which is later converted into values between 0 and 0.02.
        
        '''

        # Convert val into a float between 0-0.02
        q_lvl = val/200
        
        # create a copy from the min_eigen_val results so that the track bar updates properly
        harris_corner_detection = np.copy(min_eigen_val)
        harris_corner_detection[harris_corner_detection <= q_lvl] = 0

        # Make another copy.. used to for the non maximum supression
        harris_unaltered = np.copy(harris_corner_detection)

        harris_corner_detection[harris_corner_detection > q_lvl] = 1 # This contains the final result for STEP 3

        # Non-Maximum Supression
        filter_size = 10 # CHANGE FILTER SIZE HERE
        data_max = filters.maximum_filter(harris_unaltered, filter_size)
        maxima = (harris_unaltered == data_max)
        data_min = filters.minimum_filter(harris_unaltered, filter_size)
        diff = ((data_max - data_min) > q_lvl)
        maxima[diff == 0] = 0

        # Apply the circles of the non-maximum supression results
        non_max_suppression = np.array(maxima, dtype=min_eigen_val.dtype)

        src_copy = np.array(src, copy=True) # Create Copy to properly update the image 
        for coordinates in zip(*np.where(non_max_suppression > 0)):
            cv2.circle(src_copy, coordinates[::-1], 4, (0, 200, 0), cv2.FILLED)
        

        # Convert the Gray Scale Harris "Blob" Image into the same type as the src image
        # This allows us to display both images
        harris_corner_detection= cv2.cvtColor(harris_corner_detection,cv2.COLOR_GRAY2RGB)
        harris_corner_detection*=255
        harris_corner_detection = np.uint8(harris_corner_detection)

        # Stack the images for both steps 3 & 4
        img_stack = cv2.vconcat([harris_corner_detection, src_copy])
        cv2.imshow(title, img_stack)


    cv2.namedWindow(title)
    cv2.createTrackbar('Threshold', title, quality_lvl, max_quality_lvl, on_trackbar)
    on_trackbar(quality_lvl)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()