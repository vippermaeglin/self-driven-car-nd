# **Finding Lane Lines on the Road** 

## Writeup

### by Vinícius Arruda

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied a Gaussian Blur to remove noises and extracted the edges using the Canny Detector algorithm. Finally I defined a 4 side polygon as my region of interest and used the Hough Transformation. 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function to calculate an averaged Slope and Intercept coeficients to right and left lines respectivaly. I calculated the average for each point and used the polyfit function to achieve these coefivients, and then I applied them to the first and last point estimated on each lane.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen if the lanes are curving (like the challenge).


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to adjust the polymonial fit to curves, or another simplest solution to such situation would be to reduce the line detection to ignore curves at certain distance.
