**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set
  of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image_undist]: ./img/writeup/calibration_checkerboard.jpg "Undistorted checkerboard"
[image_car_undist]: ./img/writeup/calibration_car.jpg "Undistorted car capture image"
[image_thresh]: ./img/writeup/thresholding.jpg "Thresholding pipeline"
[image_perspective]: ./img/writeup/warping.jpg "Perspective warping"
[image_hist]: ./img/writeup/hist.jpg "Histogram of lower part of binary mask"

[image_lane_finding]: ./img/writeup/lane_finding.jpg "Visualization of the lane line finding algorthitm"
[image_full]: ./img/writeup/full_output.jpg "Full pipeline effect"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.
You're reading it!

###Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step can be seen in `undistort.py`.

The main export of this module is `undistort_factory` function,
that conveniently wraps preparing the undistortion function.

We start by computing the undistortion parameters.
We prepare "object points", which will be the (x, y, z) coordinates
of the chessboard corners in the world. Here I am assuming the chessboard is
fixed on the (x, y) plane at z=0, such that the object points are the same for
each calibration image.  Thus, `objp` is just a replicated array of
coordinates, and `objpoints` will be appended with a copy of it every time I
successfully detect all chessboard corners in a test image.  `imgpoints` will
be appended with the (x, y) pixel position of each of the corners in the image
plane with each successful chessboard detection.  

We then use the output `objpoints` and `imgpoints` to compute the camera
calibration and distortion coefficients using the `cv2.calibrateCamera()`
function. These parameters are stored in pickle file for later reuse.

The `cv2.undistort()` distortion correction functionality is wrapped into
`undistort_factory` function that loads saved undistortion parameters and
returns the final undistorting function.  Application of this function to one
of the supplied checkerboard calibration images can be seen below:
![alt text][image_undist]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Applying the above described undistortion produced by `undistort_factory` from 
`undistort.py`, yields the result visible below. The effect arguably is subtle, but important.
![alt text][image_car_undist]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


I used a combination of color and gradient thresholds to generate a binary
image. The implementation is contained in the file `color_grad.py` and has the
following steps:
  - thresholding the abs value of Sobel derivative operator in x direction
  - thresholding the quadratic norm of Sobel derivative operator
  - thresholding the direction of two-dimensional Sobel derivative operator
  - thresholding the red color and saturation (from HSL model)

These steps are showcased in the below illustration.
![alt text][image_thresh]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is centered around a function
`make_warpers` in file `perspective.py`. This function prepares and returns two
warping functions: `warp` and `unwarp`. This functions are each others inverse
and `warp` maps from the standard vehicle capture image to the 'birds view'
image.

The transformation is the standard perspective warp, defined by mapping of the
following points in a 1280x720 image:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 700        | 100, 710      | 
| 515, 472      | 100, 10       |
| 764, 472      | 1180, 10      |
| 1280, 700     | 1180, 710     |

I verified that my perspective transform was working as expected by drawing the
`src` and `dst` points onto a test image and its warped counterpart to verify
that the lines appear parallel in the warped image. This can be seen in the
below image.
![alt text][image_perspective]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The whole file `finding_lines.py` is dedicated to identifying lane-line pixels.

The flow of finding a good starting line fit is as follows
  1. finding a good starting point by computing a histogram of the lower half of the 
    binary input iamge. Identify the two peaks of it and treat them as starting x values.
![alt text][image_hist]
  2. moving up the image, put a box around the starting xs. Mark the 1 pixels in the box as hot.
  3. recenter the box around the hot pixels (if there is enough of them)
  4. if there is still place up the image, jump to pt 2 above

At the end of the above iteration take all of the hot pixels and fit 2nd degree
polynomial line to them.

The full action of the algorithm on a sample image is shown below.

![alt text][image_lane_finding]

To make the predictions more stable I use the fact that lane lines move smoothly in time
in the observed image. Therefore I expotentially smooth my detections over time. This
is implemented in file `process_video.py` in lines 20-23.


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I have prepared a function that based on the final left and right lane-line fit 
draws the lane on the (undistorted) input image. This function can be seen in file
`finding_lines.py` in lines 228 - 252.

![alt text][image_full]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./out/project_out.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked
and why, where the pipeline might fail and how I might improve it if I were
going to pursue this project further.  

