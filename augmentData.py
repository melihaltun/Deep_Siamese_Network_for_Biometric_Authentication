#
# @file augmentData.py
# @author Melih Altun @2023
#

# Applied data augmentation and downsampling to the Sapienza University Mobile Palmprint Database:SMPD,
# Increases the number of images, downsamples and standardizes the size of images.
# https://www.kaggle.com/datasets/mahdieizadpanah/sapienza-university-mobile-palmprint-databasesmpd

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# augment data and save in a standard size for all images

def plotImages(images_arr, numImages):
    fig, axes = plt.subplots(1,numImages, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def resizeImage(img):
    output_ln = 408
    output_wd = 306
    [ln, wd, _] = img.shape
    if ln / wd > 1.3 and ln / wd < 1.4:
        img = cv2.resize(img, (output_wd, output_ln), interpolation=cv2.INTER_LINEAR)
    elif ln / wd > 1.7 and ln / wd < 1.82:
        bottom_crop = 0.05
        top_crop = 0.2
        top_rows = int(top_crop * ln)
        bottom_rows = int(bottom_crop * ln)
        cropped_img = img[top_rows:ln - bottom_rows, :]
        img = cv2.resize(cropped_img, (output_wd, output_ln), interpolation=cv2.INTER_LINEAR)
    return img


processed_folder = 'D:/palm_print_proj/processed_palm_print_db'   # folder with image inputs
augmented_folder = 'D:/palm_print_proj/augmented_palm_print_db'   # folder with image outputs

gen = ImageDataGenerator(rotation_range=5, width_shift_range=0.01, height_shift_range=0.01, shear_range=0.12, zoom_range=0.06, channel_shift_range=10, horizontal_flip=False)
numExtraImages = 3

if os.path.isdir(augmented_folder) is False:
    os.makedirs(augmented_folder, exist_ok=True)
    for folder in glob.glob(os.path.join(processed_folder, '*')):
        if os.path.isdir(folder):
            delimiter = '\\'
            substrs = folder.split(delimiter)
            folderName = substrs[-1]
            for file in glob.glob(os.path.join(folder, '*')):
                if os.path.isfile(file):
                    print('Processing file ' + file)
                    delimiter = '\\'
                    substrs = file.split(delimiter)
                    fileName = substrs[-1]
                    assert os.path.isfile(file)

                    img = cv2.imread(file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = resizeImage(img)
                    image = np.expand_dims(img, 0)
                    # plot original image
                    #plt.imshow(image[0])

                    aug_iter = gen.flow(image)
                    aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(numExtraImages)]
                    # plot augmented images
                    #plotImages(aug_images, numExtraImages)

                    fileModified = fileName[:6] + '0' + fileName[6:]
                    if os.path.isdir(augmented_folder+'/'+folderName) is False:
                        os.makedirs(augmented_folder+'/'+folderName)

                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(augmented_folder + '/' + folderName + '/' + fileModified, img)

                    for k in range(1, np.min([numExtraImages+1, 10])):
                        fileModified = fileName[:6]+chr(k+48)+fileName[6:]
                        img = cv2.cvtColor(aug_images[k-1], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(augmented_folder+'/'+folderName+'/'+fileModified, img)
