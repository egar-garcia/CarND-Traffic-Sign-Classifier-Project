**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data_summary.png "Data Visualization"
[image2]: ./new_images_report.png "5 new images"
[image3]: ./new_images_predictions.png "Predictions for the 5 new images"
[image4]: ./new_images_top_probability_predictions.png "Top 5 predictions for the 5 new images"
[image5]: ./feature_map_conv1.png "Network's feature maps visualizations - Convolutional Layer 1"
[image6]: ./feature_map_conv2.png "Network's feature maps visualizations - Convolutional Layer 2"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/egar-garcia/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate the summary statistics of the traffic signs data set, to get the information points bellow. I specially used the 'shape' attribute to get the size of the different sets and shape of images. For getting the number of categories, the strategy was for each data set to create a set containing the labes in it and getting the maximum for all the three sets.

* The number of training examples (for this project 34799)
* The number of validation examples (for this project 4410)
* The number of testing examples (for this project 12630)
* The image data shape (for this project 32x32)
* The number of classes (labels) (for this project 43)

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It includes:
* A sample of images from each one of the data sets (training, validation and test)
* A pie graph indicating percentage of the total of images and size of each one of the data sets
* A bar grap indicating the number of images per classification/label and per each data set
* A sample image for each one of the classifications/labels

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Originally the images are a matrix of pixels with three components (red, green and blue) each one an integer in the range from 0 to 255, so the first step was to scale the color components to floating point numbers in the range from -1.0 to 1.0, this with the purpose of the data to have mean zero and equal variance.

A suggestion was to transform the images to grayscale in order to identify the figure patters of the images, however some of the trafic sign images seem to be dependant on their color, so the solution I decided to take was to add an extra (color) layer for each pixel which would be the the black-and-white component using Luma transformation, in this way the images of shape 32x32x3 where transformed to a shape 32x32x4. When doing the practical tests, it seamed that with this kind of transformation the trinning could achieve higher accuracy rates than just using B/W images or 3-component color images.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc. Consider including a diagram and/or table describing the final model.

My final model way based in LeNet5, which consisted of the following layers:

| Layer         		|     Description	        					               |
|:---------------------:|:------------------------------------------------------------:| 
| Input         		| 32x32x4 image (3 color componets and 1 B/W component)        |
| Convolution 5x5     	| inputs 32x32x4, 1x1 stride, valid padding, outputs 28x28x32  |
| RELU					| inputs 28x28x32, outputs 28x28x32							   |
| Max pooling	      	| inputs 28x28x32, 2x2 stride, outputs 14x14x32 		       |
| Convolution 5x5     	| inputs 14x14x32, 1x1 stride, valid padding, outputs 10x10x64 |
| RELU					| inputs 10x10x64, outputs 10x10x64							   |
| Max pooling	      	| inputs 10x10x64, 2x2 stride, outputs 5x5x64 				   |
| FLATEN		        | inputs 5x5x64, outputs 1600        						   |
| Fully connected		| inputs 1600, outputs 1200                                    |
| RELU                  | inputs 1200, outputs 1200                                    |
| Fully connected		| inputs 1200, outputs 1024                                    |
| RELU                  | inputs 1024, outputs 1024                                    |
| Fully connected		| inputs 1024, outputs 43                                      | 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, as based in the LeNet lab, I decided to use Adam optimizer with alearning rate of 0.001. For the batch size I finilly used 128, I tried to use different batch sizes like 256, 512, 1024, 2048, 4096, however they were presenting lower accuracy as long as the epochs progress and having problems to go beyond an accuracy rate of 0.9, the one that presented more progress on accuracy rate was of size 128. For the training rate other values were tested like 0.01, 0.05 and 0.0001 which maked the accuracy to be low or progressing too slow.

In terms of the number of epochs, I actually took an approach of early termination, taking a minimun accuracy rate of 0.945 to reach in order to conclude the training and a maximum of 50 epochs, the idea is if the minimum accuracy is reached and the training doesn't seem to be improving the accuracy, then the training is terminated. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The strategy was to continue the training until a minimum desired accuracy was reached, if the accuracy was improving then the training continued, else the training was terminated. The minimum accuracy rate of 0.945 on the validation set was choosen, because in practical terms when the training model was applied to the tests set this was a bit lower (around 0.150 less) than in the validation set. During the tests the accuracy of 0.945 was reached between epochs 10 and 35, in the reported example the accuracy rate on the validation set was 0.952 and when applied to the test set was 0.938.

My final model results were:
* Validation set accuracy of 0.952 
* Test set accuracy of 0.938

If a well known architecture was chosen:
* What architecture was chosen? R: LeNet 5
* Why did you believe it would be relevant to the traffic sign application? R: LeNet is a well know architecture to be applied on simbols' recognition apearing on images
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? During the training the model was capable of reaching accuracy rates higher than 0.93, which accomplished the the specific requirements.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2]

The third one 'double curve' seems to be difficult to clissify, because its general shape is similar to many other signs and it cointains line paters that are difficult to differentiate from other similar images like dangerous curves, slippery road or wild animals crossing. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image3]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.8%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For all the images, the model is pretty much sure that the predictions for the image classes match with the actual ones, these are the results of the top 5 predictions for each one of the images:

![alt text][image4]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The following one is the visualization of the first convolutional layer, it looks like the model is identifying some basic shapes like lines, circles and semi-circles.

![alt text][image5]


The following one is the visualization of the second convolutional layer, it looks like at this stage the model still identifies some line shapes, however most of the data doesn't seem to have any sense for human/logic interpretation.

![alt text][image6]
