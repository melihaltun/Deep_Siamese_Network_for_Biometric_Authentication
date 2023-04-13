# Deep_Siamese_Network_for_Biometric_Authentication
Deep Siamese convolutional neural net for biometric authentication of palm print images

This code performs biometric authentication using two palm images. One of the images is considered as template and the second one is input. If the template and the input match each other the model outputs 1 and if they are images from different people, the model returns 0. 

![image](https://user-images.githubusercontent.com/40482921/231610660-1283e705-2118-43a7-8b1f-f9ad89369b6d.png)

The model is trained with Sapienza University palm image dataset from Kaggle: 
https://www.kaggle.com/datasets/mahdieizadpanah/sapienza-university-mobile-palmprint-databasesmpd 

Once the dataset is downloaded, modify the input and output paths in each source file as desired.

There are five Python script files each with a specific task. They should be executed with the order below:

1) selectPalmPrints.py: Goes through the entire dataset and picks the top view palm images. The image orientation is arbitrary in the database. This script also applies some image processing to make sure all images are in the same orientation. 
2) augmentData.py: Resizes each selected image and applies data augmentation by generating variations of each image.
3) trainTestValSplit.py: Goes through the augmented data set and splits them into training, validation and test images.
4) getTrainingStats.py: Extracts mean RGB value of the training set, which is used in the pre-processing before model training.
5) trainModel.py: Forms the training validation and test sets from by combining each set into matching and non-matching pairs. Defines the Siamese neural network model, and trains and tests the model. 

The model achieves 96.0% accuracy with a precision rate of 93.8% and a recall rate of 98.6% after 48 epochs. 
Utilizing GPU is recommended. Due to model complexity, a single epoch with CPU may last well over an hour. On the other hand, even a modest graphics card can speed up the training by 15 to 20 times. 

A combination of Tensorflow 2.10.0, CUDA 11.2, CUDNN 8.8.1 and Zlib works for this model, although other configurations may also be possible. 
