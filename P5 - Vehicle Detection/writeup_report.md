# **Vehicle Detection & Tracking** 

## Writeup

### by Vinícius Arruda

---
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/image01.png
[image2]: ./examples/image02.png
[image3]: ./examples/image03.png
[image4]: ./examples/image04.png
[image5]: ./examples/image05.png
[video1]: ./video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

My project includes the following files:
* [P5.ipynb](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P5%20-%20Vehicle%20Detection/P5.ipynb) containing the jupyter notebook with my code
* [video.mp4](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P5%20-%20Vehicle%20Detection/video.mp4) containing the final video result 
* [writeup_report.md](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P5%20-%20Vehicle%20Detection/writeup_report.md) summarizing the results

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Notebook In[2]: I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image5]

#### 2. Explain how you settled on your final choice of HOG parameters.

Notebook In[2]: I tried various combinations of parameters and this was my final result:

| Parameter           |Value	  |  Description						|
|:-------------------:|:---------:|:-----------------------------------:|
|  color_space    |  'YCrCb'   |  Can be RGB, HSV, LUV, HLS, YUV, YCrCb |
|  orient         | 9          | HOG orientations
|  pix_per_cell   | 8          | HOG pixels per cell
|  cell_per_block | 2          | HOG cells per block
|  hog_channel    | 'ALL'      | Can be 0, 1, 2, or "ALL"
|  spatial_size   | (32, 32)   | Spatial binning dimensions
|  hist_bins      | 32         | Number of histogram bins
|  spatial_feat   | True       | Spatial features on or off
|  hist_feat      | True       | Histogram features on or off
|  hog_feat       | True       | HOG features on or off

I also tried different color spaces, but YCrCb provided the best result. Increasing the ```orientation``` enhanced the accuracy of the classifier, but increased computational time as well.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Notebook In[5]: First I fed the LinearSVC model with the extracted features using default settings, the trained model had accuracy of 99.35% on test data. Then the trained model and parameters used for training were saved to pickle files to be further used by vehicle detection pipeline.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Notebook In[3]: I decided to use a multi-scale window approach for a higher coverage of potential detections. It's also prevents calculation of feature vectors for the entire image and thus helps in speeding.

| Window 1      | Window 2      | Window 3      |
|:-------------:|:-------------:|:-------------:|
| ystart = 380  | ystart = 400  | ystart = 500  |
| ystop = 480   | ystop = 600   | ystop = 700   |
| scale = 1     | scale = 1.5   | scale = 2.5   |

The figure below shows the multiple scales under consideration overlapped on image.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Notebook In[5, 6, 7, 8]: Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I saved the positions of true detections in each frame of the video. From the true detections I created a heatmap and then thresholded that map to identify vehicle positions: 

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Notebook In[10]: Here's a [link to my video result](./video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall the implemented classification model is quite simple and it can be optimized to provide higher accuracy. Also, this pipeline may have problems in difficult shading/illumination conditions and the multi-window search may be optimized further for better speed.

