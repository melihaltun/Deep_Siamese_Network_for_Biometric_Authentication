#
# @file selectPalmPrints.py
# @author Melih Altun @2023
#

# Selects top view images from the Sapienza University Mobile Palmprint Database:SMPD,
# applies a noise filter and processes them to have the same orientation.
# https://www.kaggle.com/datasets/mahdieizadpanah/sapienza-university-mobile-palmprint-databasesmpd

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2
from scipy.signal import filtfilt, butter, find_peaks


def rotateImg(img, angle):
    center = tuple(np.array(img.shape[1::-1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation matrix to the input image
    rotated = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated


def plotTwoCurves(x, y1, y2):
    fig, ax = plt.subplots()
    ax.plot(x, y1, color='red', label='top')

    # Plot the second curve (in red) on the same axis
    ax.plot(x, y2, color='blue', label='bottom')

    # Set the axis labels and title
    ax.set_xlabel('x (pixel pos''n)')
    ax.set_ylabel('intensity')
    ax.set_title('top & bottom image intensity')
    # Add a legend to the plot
    ax.legend()
    # Display the plot
    plt.show()


raw_data_folder = 'D:/palm_print_proj/palm_print_db'    # folder with image inputs
processed_folder = 'D:/palm_print_proj/processed_palm_print_db'    # folder with image outputs

#fiter_coeffs
f_b, f_a = butter(2, 1/10, btype='lowpass', analog=False)

if os.path.isdir(processed_folder) is False:
    os.makedirs(processed_folder)
    for folder in glob.glob(os.path.join(raw_data_folder, '*')):
        if os.path.isdir(folder):
            delimiter = '\\'
            substrs = folder.split(delimiter)
            folderName = substrs[-1]
            #if folderName != '044':     #in case you want to work on a specific class
            #    continue
            for file in glob.glob(os.path.join(folder, '*')):
                if os.path.isfile(file) and "_F_" in os.path.basename(file):  #only process top view images
                    print('Processing file '+file)
                    delimiter = '\\'
                    substrs = file.split(delimiter)
                    fileName = substrs[-1]
                    img = cv2.imread(file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.medianBlur(img, 5)  # median filter image
                    [ln, wd, _] = img.shape
                    #make sure the images all have the same alignment
                    if wd > ln:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        [ln, wd, _] = img.shape
                    gr_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    rd_img = img[:,:,1]

                    # check the intensity of top and bottom parts of the image
                    topStrip = rd_img[1:ln//55, :]
                    topStrip = np.mean(topStrip, axis=0)/255
                    bottomStrip = rd_img[54*ln//55:, :]
                    bottomStrip = np.mean(bottomStrip, axis=0)/255

                    topStrip = filtfilt(f_b, f_a, topStrip)
                    bottomStrip = filtfilt(f_b, f_a, bottomStrip)

                    # count the number of peaks. Fingers generate multiple peaks and wrist generates a single peak
                    minTop = np.min(topStrip)
                    maxTop = np.max(topStrip)
                    minBottom = np.min(bottomStrip)
                    maxBottom = np.max(bottomStrip)
                    diffTop = maxTop-minTop
                    diffBottom = maxBottom-minBottom
                    avgDiff = diffTop/2 + diffBottom/2
                    topStrip = np.pad(topStrip, (1, 1), mode='constant', constant_values=minTop)
                    bottomStrip = np.pad(bottomStrip, (1, 1), mode='constant', constant_values=minBottom)

                    count = 0
                    peaksTop = np.array([0])
                    peaksBottom = np.array([0])

                    #plotTwoCurves(range(wd+2), topStrip-np.mean(topStrip), bottomStrip-np.mean(bottomStrip))

                    # make sure the fingers are on top. If not rotate image
                    while count < 3 and  len(peaksTop) == len(peaksBottom):
                        peaksTop, _ = find_peaks(topStrip, prominence=avgDiff/(3+count), distance=wd//9, width=wd//20)
                        peaksBottom, _ = find_peaks(bottomStrip, prominence=avgDiff / (3+count), distance=wd//9, width=wd//20)
                        if len(peaksTop) < len(peaksBottom):
                            img = cv2.rotate(img, cv2.ROTATE_180)
                            break
                        count += 1

                    if len(peaksTop) == len(peaksBottom):
                        darkBottom = bottomStrip < diffBottom*0.1 + minBottom
                        darkTop = topStrip < diffTop*0.1 + minTop
                        if np.count_nonzero(darkBottom) < np.count_nonzero(darkTop):
                            img = cv2.rotate(img, cv2.ROTATE_180)

                    # save processed image
                    savePath = processed_folder + '/' + folderName + '/'
                    if os.path.isdir(savePath) is False:
                        os.makedirs(savePath)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(savePath + fileName, img)
