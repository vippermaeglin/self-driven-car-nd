# **Behavioral Cloning** 

## Writeup

### by Vinícius Arruda

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P3%20-%20Behavioral%20Clonning/model.py) containing the script to create and train the model
* [drive.py](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P3%20-%20Behavioral%20Clonning/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P3%20-%20Behavioral%20Clonning/model.h5) containing a trained convolution neural network 
* [writeup_report.md](https://github.com/vippermaeglin/self-driven-car-nd/blob/master/P3%20-%20Behavioral%20Clonning/writeup_report.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48 (model.py lines 102-138) 

The model includes RELU layers to introduce nonlinearity and a few ELU layers for a higher classification accuracy, and the data is normalized in the model using a Keras lambda layer (code line 104). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 127). 

The model was trained and validated on different data sets to ensure that the model was not overfitting, and apllied Adam optimization at the end (code line 141-147). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam Optimizer, so the learning rate was not tuned manually (model.py line 145).

#### 4. Appropriate training data

The training data was genrated by me using the mouse as controller on the first track of Udacity Simulator. Afte I used a random combination of center/left/right cameras on my model to train it in different lanes perspectives. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

My architecture was based on the neural network used by NVIDIA with a few modifications.  
At first, I used the NVIDIA neural network with all 5 CNN layers, but it doesn't work well. As suggested by Udacity, I added a lambda layer to normalize data, Adam Optimizer to improve the weight updates and preprocessed the data before training. I also simplified the network to reduce my tests duration by removing 2 CNN layers and decreasing convolutions depths.  
Here's my final architecture:

| Layer         		|     Description	        			                                  		|
|:-----------------:|:-------------------------------------------------------------------:|
| INPUT         		| Lambda layer with cropping                                          |
| CNN 1             | 5x5 filter, stride of 2, 24 outputs (depth) and RELU activation     |
| CNN2              | 5x5 filter, stride of 2, 36 outputs (depth) and RELU activation     |
| CNN3              | 5x5 filter, stride of 2, 48 outputs (depth) and ELU activation      |
| MAX POOLING       | 2x2 filter                                                          |
| FLATTEN           | Flatten layer                                                       |
| FULLY CONNECTED 1 | Input 100 and ELU activiation                                       |
| DROPOUT           | 50% rate                                                            |
| FULLY CONNECTED 2 | Input  50 and ELU activiation                                       |
| FULLY CONNECTED 3 | Input  10 and ELU activiation                                       |
| OUTPUT            | Output with trained weighted and biases                             |
