# **Traffic Sign Recognition** 

## Writeup

### by Vinícius Arruda

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/image1.png "Visualization"
[image2]: ./examples/image2.png "Distribution"
[image3]: ./examples/image3.png "Normalized"
[image4]: ./examples/image4.png "New Ditribution"
[image5]: ./examples/image5.png "Traffic Sign 1"
[image6]: ./examples/image6.png "Traffic Sign 2"
[image7]: ./examples/image7.png "Traffic Sign 3"
[image8]: ./examples/image8.png "Traffic Sign 4"
[image9]: ./examples/image9.png "Traffic Sign 5"
[image9]: ./examples/image10.png "Traffic Sign 6"
[image11]: ./examples/image11.png "Predictions"
[image12]: ./examples/image11.png "Accuracy"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submission

- Link to my [jupyter notebook](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P2%20-%20Traffic%20Sign%20Classifier/Traffic_Sign_Classifier.ipynb)
- Link to my [HTML result](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P2%20-%20Traffic%20Sign%20Classifier/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Notebook In[1]: I loaded the data using pickle.  
Notebook In[2]: I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Notebook In[3]: Here is an exploratory visualization of the data set showing 8 random images/labels.
![alt text][image1]

Notebook In[4]: Then I plotted the occurrence of each image class to show how the data is distributed for each data set.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Notebook In[5]: As a first step, I decided to convert the images to grayscale because I read a few papers with great results using this technique and I don't really think colors are relevant to traffic signs (most of them are mono symbols).  
Notebook In[6]: To add more data to the the data set, I created additional images for training through randomized modifications like small rotations or opencv affine.  
As a last step, I normalized the gray pixel values to 128 to improve the speed of training.  
  
Here is an example of a new image inserted and the original grayscale image used to create it(on the right):

![alt text][image3] 

I increased the train dataset size to 89860 and the validation set gained 20% of this original total mentioned. I did this using train_test_split method from scikit learn library. Now no image class in the train set has less then 1000 images.

![alt text][image4] 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Notebook In[7]: My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   					|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        					|
| RELU					|												|
| Dropout				| 50% keep        								|
| Fully connected		| input 122, output 84        					|
| RELU					|												|
| Dropout				| 50% keep        								|
| Fully connected		| input 84, output 43        					|

 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Notebook In[8]: To train the model, I used an LeNet for the most part that was given, but I did add an additional convolution without a max pooling layer after it like in the udacity lesson.  I used the AdamOptimizer with a learning rate of 0.00097.  The epochs used was 27 while the batch size was 156.  Other important parameters I learned were important was the number and distribution of additional data generated.  I played around with various different distributions of image class counts and it had a dramatic effect on the training set accuracy.  It didn't really have much of an effect on the test set accuracy, or real world image accuracy.  Even just using the default settings from the Udacity lesson leading up to this point I was able to get 94% accuracy with virtually no changes on the test set.  When I finally stopped testing I got 94-95.2% accuracy on the test set though so I think the extra data improved training accuracy, but not a huge help for test set accuracy.  Although this did help later on with the images from the internet.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Notebook In[8]: The accuracy was calculated using the "evaluate" function and then plotted a chart with all epochs:
![alt text][image12]

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.8% 
* test set accuracy of 94.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?  
I started with the original LeNet code from lessons exercises with default parameters but both accuracy was below 93%.
* What were some problems with the initial architecture?  
The main issues was the poor normalization and wrong parameters (underfitting).
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.  
After increasing the data set on normalization and changing hyperparameters I still couldn't reach 93% very easily, so I added another convolution and dropouts with a 50% probability to achieve a satisfactory result.
* Which parameters were tuned? How were they adjusted and why?  
Epoch, learning rate, batch size, and drop out probability were all parameters tuned along with the number of random modifications to generate more image data was tuned. For Epoch the main reason I tuned this was after I started to get better accuracy early on I lowered the number once I had confidence I could reach my accuracy goals. The batch size I increased only slightly since starting once I increased the dataset size. The learning rate I think could of been left at .001 which is as I am told a normal starting point, but I just wanted to try something different so .00097 was used. I think it mattered little. The dropout probability mattered a lot early on, but after awhile I set it to 50% and just left it. The biggest thing that effected my accuracy was the data images generated with random modifications. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  
The convolutional and dropout layers was basically my architectural modifications after reading several papers and articles at Medium. Dropout was amazing to drastically improve generalization of my model and preventing overfitting.

### Test a Model on New Images

#### 1. Choose five or more German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Notebook In[9]: Here are six German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10]

The fourth image might be difficult to classify because an empty sign would be interpreted as an unfocused image (poor quality).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Notebook In[10,11,12]: Here are the results of the prediction:

![alt text][image11] 


The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set although I did throw it a softball.

#### 3. Describe how certain the model is when predicting on each of the six new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Notebook In[10,11,12]:The probabilities for each prediction is available in the above image.

