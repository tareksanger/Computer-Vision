import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def main():
    img_dir = os.path.abspath('img/')

    img1 = cv2.imread(os.path.join(img_dir, 'uttower_right.jpg'), 0)            # queryImage
    img2 = cv2.imread(os.path.join(img_dir, 'large2_uttower_left.jpg'), 0)      # trainImage

    detector = cv2.AKAZE_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)


    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Draw first 20 matches.
    matches_img = np.zeros((1, 2))
    matches_img = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20], outImg=matches_img, flags=2)
    cv2.imshow('Matches', matches_img)

    src_pts = np.float32([ kp1[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)

    # Compute the Homography & warp img2
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    warped = cv2.warpPerspective(img1, H, img2.shape[::-1]) # warped image

    # Merge the images
    result = np.maximum(warped, img2)

    # Display the 2 Images!
    cv2.imshow("Warped", warped)
    cv2.imshow('Merged', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()