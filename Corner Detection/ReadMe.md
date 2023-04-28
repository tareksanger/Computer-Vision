## Packages:

- OpenCV
- Numpy
- SciPy

## Harris Corner Detection

The code for Part 1 can be found in `harris_corner_detection/__main__.py`. To run the Corner Detection tests simply call.

```sh

python harris_corner_detection

```

In Harris Corner Detection the track bar has 5 options- {0, 1, 2, 3, 4}. This values are divided by 200 such that the max value is 0.02.

Note that for the Non-Maximum Suppression I used a 10x10 matrix.

All result images can be found in the harris_corner_detection directory.

## Image stitching

The code for Image stitching can be found in `image_stitching/__main__.py`. To run the test for image_stitching simply call.

```sh

python image_stitching

```

All result sample images can be found in the Part2.

### What Could we do in place of the OR operation to reduce these anomalies?

To clear the anomalies and increase the quality of the image, we can start by taking the average of each pixel and their brightness value. Unfortunately, this doesn't completely solve the issue even though it does give us a slightly better image. What we're left with after the averaging are still hard lines where the images overlap, we can further improve this by blending. How do we blend? This is down using a weighting function. This weighting function calculates the weight of the pixel by also calculating its distance the the nearest boundary point - An example of such function is the bwdist in MATHLAB. https://www.youtube.com/watch?v=D9rAOAL12SY
