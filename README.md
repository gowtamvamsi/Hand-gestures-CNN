# Hand gestures using webcam and CNN (Convoluted Neural Network)

## Introduction
If you have a brief understanding of deep learning, convolutional neural networks and wants to apply CNN to an Image classification problem, I hope this post will be helpful to you.

The goal of this project is to accurately classify images of different hand gestures like thumbs up, showing numbers, etc. This is a supervised classification problem, So we can use either simple classification machine learning algorithms or Deep learning CNN. In this article, we are going to use CNN's to solve this problem.

We will write 3 python codes. First for generating data, the second one for the training model and the last one for classifying hand gestures using the already generated CNN model and image from the webcam.
Python packages required :
numpy, PIL(Pillow), Open CV (opencv-python), Keras (with TensorFlow)

Machine learning project life cycle:
Collect the data/Load the data/Generate the data
Build the model
Train the model
Test the model
If the data is temporal, it is best to train the model on recent data within a period.
