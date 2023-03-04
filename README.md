# Face_Mask_Detection
This repository contains a micro project entitled "Real Time Face Mask Detection using VGG16 CNN".

# Conda Virtual Environment
* conda create --name <environment_name> python=3
* conda activate <environment_name>
* conda install -c anaconda cmake pkg-config
* conda install -c conda-forge gstreamer=1.14.5 gst-plugins-base=1.14.5 gst-plugins-good=1.14.5
* conda install -c conda-forge opencv
* conda install tensorflow keras
* conda install matplotlib
* conda install numpy
* conda install pandas
* conda install scikit-learn

# Introduction
The project aims to detect whether a person is wearing a face mask or not using computer vision and machine learning techniques. The project utilizes the VGG16 pre-trained deep learning model and a custom dataset consisting of images with and without masks. The project is implemented using the Keras deep learning framework, OpenCV, and Python.

# Dataset
Dataset consists of 7553 RGB images in 2 folders as with_mask and without_mask. Images are named as label with_mask and without_mask. Images of faces with mask are 3725 and images of faces without mask are 3828. This dataset is taken from following link: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset . The custom dataset is created by iterating through all the images in the respective directories using the OpenCV library. The images are resized to 224x224 pixels, as required by the VGG16 model. The dataset is then shuffled randomly to reduce any bias during training.

# Model Architecture
The pretrained VGG16 model is used as the base model for this project. The last layer of the VGG16 model is replaced with a Dense layer with a sigmoid activation function. The Dense layer consists of one neuron as the output is binary, i.e., whether the person is wearing a mask or not. The weights of the VGG16 model are frozen to avoid overfitting.

<img width="701" alt="model" src="https://user-images.githubusercontent.com/115887529/222911838-e94a9a7c-66ae-4fce-a0d9-2d5a68ba91a1.png">

# Training
The model is trained using the binary cross-entropy loss function and the Adam optimizer. The model is trained for five epochs, and the validation data is used to evaluate the performance of the model. The train_test_split function from the scikit-learn library is used to split the dataset into training and testing sets.

<img width="1121" alt="Training" src="https://user-images.githubusercontent.com/115887529/222912049-c9ecf254-e847-41cc-b633-b7fc4fb5a767.png">

# Face Detection
The project utilizes the HaarCascade classifier from OpenCV to detect faces in the input image. The detectMultiScale method is used to detect faces in the grayscale image. The coordinates of the faces are then used to draw a rectangle around the detected face in the original image.

# Testing
The project is tested using a video feed from the webcam. The model predicts whether a person is wearing a mask or not based on the input image. The predicted result is then displayed on the video feed, along with a rectangle around the detected face.

# Conclusion
The project demonstrates the use of computer vision and machine learning techniques to detect face masks in real-time. The project utilizes the VGG16 model for feature extraction and the HaarCascade classifier for face detection. The project can be further improved by using other pre-trained models or by training a custom model on a larger dataset.
