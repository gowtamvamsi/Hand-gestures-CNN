# Hand gestures using webcam and CNN (Convoluted Neural Network)

## Introduction
If you have a brief understanding of deep learning, convolutional neural networks and wants to apply CNN to an Image classification problem, I hope this post will be helpful to you.

The goal of this project is to accurately classify images of different hand gestures like thumbs up, showing numbers, etc. This is a supervised classification problem, So we can use either simple classification machine learning algorithms or Deep learning CNN. In this article, we are going to use CNN's to solve this problem.

We will write 3 python codes. First for generating data, the second one for the training model and the last one for classifying hand gestures using the already generated CNN model and image from the webcam.
<br>
Python packages required : numpy, PIL(Pillow), Open CV (opencv-python), Keras (with TensorFlow)

Machine learning project life cycle:
* Collect the data/Load the data/Generate the data
* Build the model
* Train the model
* Test the model
* If the data is temporal, it is best to train the model on recent data within a period.

## Generating the data

We can either use data from Hand Gesture Recognition Database or we can generate our data. I choose the latter because with small changes we can convert this project from Hand gesture recognition to Multi-person facial recognition. I had written generate_data.py to automate the process of generating data

generate_data.py program will generate both the train, validate data.

If you are doing the facial recognition then save the training, validation images of faces without masking.
Save the above-mentioned program and run it with the commands below for training, validation images

Commands to generate training and validation data :
<br>
Training images
* for gesture 1 images: python generate_data.py data/train/0 1000
* for gesture 2 images: python generate_data.py data/train/1 1000
* for gesture 3 images: python generate_data.py data/train/2 1000

Validation images
* for gesture 1 images: python generate_data.py data/valid/0 200
* for gesture 2 images: python generate_data.py data/valid/1 200
* for gesture 3 images: python generate_data.py data/valid/2 200
...and so on, if you want to train model on more gestures.

## Creating model
My model consists of 4 convoluted layers each connected by a max-pooling layer to the next one. After the convolution is done it is followed by a fully connected layer and finally the output layer. Every image size in the input is 128X128X3.
<br>
Check out gesture_recognition_webcam.py for the code.
<br>
The activation function in all the layers except the output layer is relu. I used the softmax activation function for the output layer because we are solving a multi-class classification problem.
I used the dropout layer to avoid running into the overfitting problem.

## Compiling the model

I used **Adam** as a gradient descent optimizer with **categorical_crossentropy** logarithmic loss function.
If you want to solve a binary classification problem you can alter loss function to **binary_crossentropy**

## Training model
Instead of feeding all the training images to CNN at once we will use Keras technique called ImageDataGenerator which will allow us to load a batch of training images to pass-through CNN at a single time.
<br>
In every epoch, CNN should see the whole training data at least once. So, we have to set steps per epoch = 47 (train_data_size/batch_size).
<br>
After training is completed we need to save the trained model locally. So, we can use it later to classify hand gestures in a live video feed.
## Testing the model using webcam and gestures
We have to load the trained model we saved locally. After loading the model successfully we will start capturing the video through webcam.
We will read frame in every loop and before predicting the gesture it contains we need to process it as we did when generating the training data.
<br><br>
CNN will expect 4D tensor as input but after processing our mask will be of 128x128 shape. Normally in CNN, we will feed RGB (3 channels)images so they will be of shape height x width x 3. So, I feed the same mask in all the 3 channels making the masked image the size of 128x128x3. After that, we used expand_dims to convert image into 4D.
<br><br>
After all the processing on the frame, we can predict the class of the gesture.
You can execute the above program locally then a video frame will popup. Point the gestures in front of the webcam and watch the terminal for the predicted class labels. After you are done press 'q' to exit from the video feed.
<br><br>
I have done a pretty good job of isolating the noise but for ideal results generate the training data and test the trained model with a dark background.
This is a good beginner project to understand how to build layers in CNN's.

