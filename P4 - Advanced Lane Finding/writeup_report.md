# **Advanced Lane Finding** 

## Writeup

### by Vinícius Arruda

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/image01.png "Example"
[image2]: ./examples/image02.jpg "Camera Calibration"
[image3]: ./examples/image03.jpg "Undistorted Image"
[image4]: ./examples/image04.jpg "Thresholds"
[image5]: ./examples/image05.jpg "LS Channels"
[image6]: ./examples/image06.jpg "Final Threshold"
[image6]: ./examples/image07.jpg "Bird's Eye"
[image6]: ./examples/image08.jpg "Histogram"
[image6]: ./examples/image09.jpg "Detected Lines"
[video1]: ./video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

My project includes the following files:
* [P4.ipynb](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P4%20-%20Advanced%20Lane%20Finding/P4.ipynb) containing the jupyter notebook with my code
* [video.mp4](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P4%20-%20Advanced%20Lane%20Finding/video.mp4) containing the final video result 
* [writeup_report.md](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P4%20-%20Advanced%20Lane%20Finding/writeup_report.md) summarizing the results

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Notebook In[2]: I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Notebook In[4]: To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Notebook In[5, 6, 7, 8]: I used a combination gradient thresholds to generate the first binary image combining Sobel Operator, Magnitude of the Gradient and Direction of the Gradient.

![alt text][image4]

Notebook In[9, 10]: Then I splitted L and S channnels and combined them into another threshold.

![alt text][image5]

Notebook In[11]: Finally I combined both thresholds above to get the final binary image.

![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Notebook In[12]: The code for my perspective transform includes OpenCV methods like `cv2.getPerspectiveTransform` and  `cv2.warpPerspective`. I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify the bird's eye effect.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Notebook In[12]: First I took a pixel histogram of the image to identify lines regions.

![alt text][image8]

Notebook In[14, 15, 16, 17, 18]: Then I implemented the Sliding Window and Fit a Polynomial to get the result bellow.

![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Notebook In[19]: I defined the `get_curvatures(binary_warped, left_fit, right_fit)` and `get_car_offset(binary_warped, left_fit, right_fit)` functions respectively.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Notebook In[20]: Here is an example of my result on a test image:

![alt text][image01]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Once again the hardest part of this project was to test the right parameters and filters that would find the lane lines for the most out of the images. Another challenge was hiding irrelevant objects from the image that could make line detection very hard, such as trees and shades. Overall my code works well with the project video but I'm aware that these fixed thresholds values used for the different filters cannot possibly work successfully in every situation. To make it more robust these parameters could be dynamically adjusted according with the current scenario and I also would improve the "handling bad frames" approach, since it's hard to recover for a long sequence like 30 frames or more. 
