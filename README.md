[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Augmented-Reality
Python and OpenCV project for AR tag detection and tracking, Image and 3D cube super-imposition

## Overview
Our objective is to develop a Detection and Tracking algorithm that identifies AR tags IDs and orientation from a video file and superimpose an image and a 3D virtual cube on the tag. The project was programmed using python 3 with OpenCV and NumPy libraries.

## Purpose

An AR Tag is a fiducial marker system to support augmented reality. They can be used to facilitate the appearance of virtual objects, games, and animations within the real world. Augmented Reality tags are widely used in Computer Vision applications.

## Homography

A homography is a perspective transformation of a plane, i.e, a reprojection of a plane from one camera into a different camera view, subject to change in the translation (position) and rotation (orientation) of the camera.

## Project Requirements/ Dependencies

OpenCV

Python3

Numpy

## Pipeline

1) Read the current frame from the video file and convert it to Grayscale and apply threshold on it.
2) Use thresholded image and find contours on the image.
3) Find the arc length and area of each contour in the image and filter based on the area to get the correct tag area.
4) Calculate Homography and Inverse Homography of the tag area w.r.t. the reference tag image.
5) Apply Perspective transform and warp the tag area out from the video frame using the Homography matrix.
6) Convert the rescaled image to Grayscale and apply threshold on it.
7) Detect the tag position and orientation through the average intensity values.
8) Calculate Homography for the overlay image.
9) Warp the image onto the original image by applying bitwise operations
10) Based on the coordinates generated from the Tag positon, the 3D virtual cube is superimpsoed on the tag area.


## Command to run code
```
python3 ar.py
```
## References

https://en.wikipedia.org/wiki/ARTag

https://opencv.org/


## Licence
The Repository is Licensed under the MIT License.
```
MIT License

Copyright (c) 2019 Charan Karthikeyan Parthasarathy Vasanthi, Nagireddi Jagadesh Nischal, Sai Manish V

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Known Issues/Bugs

Issue with varying Area of the tag: The tag was not being detected due to the tag not being in area threshold.
