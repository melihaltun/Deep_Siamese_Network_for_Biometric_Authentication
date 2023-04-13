#
# @file getTrainingSetStats.py
# @author Melih Altun @2023
#
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# get mean rgb values from all processed training images

train_dir1 = 'D:/palm_print_proj/trainTestVal/train/1'
img_files = os.listdir(train_dir1)

r_sum = g_sum = b_sum = 0
for file in tqdm(img_files):
    img = cv2.imread(os.path.join(train_dir1, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_sum += np.sum(img[:, :, 0])/255
    g_sum += np.sum(img[:, :, 1])/255
    b_sum += np.sum(img[:, :, 2])/255

[img_sz_y, img_sz_x, _] = img.shape
r_avg = r_sum/(len(img_files)*img_sz_y*img_sz_x)
g_avg = g_sum/(len(img_files)*img_sz_y*img_sz_x)
b_avg = b_sum/(len(img_files)*img_sz_y*img_sz_x)

rgb_dict = { 'r': [r_avg], 'g': [g_avg], 'b':[b_avg]}

df = pd.DataFrame(rgb_dict)
df.to_csv('mean_rgb_val.csv', index=False)
