#
# @file trainValTestSplit.py
# @author Melih Altun @2023
#

# Forms the train, validation and test sets from pre-processed palm images
# For each subject with palm prints the code selects one palm image as template to match
# it then generates 30 matching and 30 non-matching training pairs using the template and a random sample,
# 6 matching and 6 non-matching validation pairs and 3 matching, 3 non-matching test pairs.
# Overall 2760 positive 2760 negative training, 552 positive, 552 negative validation,
# 276 positive and 276 negative test data are generated.

import os
import random
import shutil

input_dir = 'D:/palm_print_proj/augmented_palm_print_db'

# Set up directories
train_dir1 = 'D:/palm_print_proj/trainTestVal/train/1'
val_dir1 = 'D:/palm_print_proj/trainTestVal/validation/1'
test_dir1 = 'D:/palm_print_proj/trainTestVal/test/1'

train_dir0 = 'D:/palm_print_proj/trainTestVal/train/0'
val_dir0 = 'D:/palm_print_proj/trainTestVal/validation/0'
test_dir0 = 'D:/palm_print_proj/trainTestVal/test/0'

train_dir_t = 'D:/palm_print_proj/trainTestVal/train/t'
val_dir_t = 'D:/palm_print_proj/trainTestVal/validation/t'
test_dir_t = 'D:/palm_print_proj/trainTestVal/test/t'

num_train = 30
num_val = 6
num_test = 3

if not os.path.exists(train_dir1):
    os.makedirs(train_dir1)
if not os.path.exists(val_dir1):
    os.makedirs(val_dir1)
if not os.path.exists(test_dir1):
    os.makedirs(test_dir1)
if not os.path.exists(train_dir0):
    os.makedirs(train_dir0)
if not os.path.exists(val_dir0):
    os.makedirs(val_dir0)
if not os.path.exists(test_dir0):
    os.makedirs(test_dir0)
if not os.path.exists(train_dir_t):
    os.makedirs(train_dir_t)
if not os.path.exists(val_dir_t):
    os.makedirs(val_dir_t)
if not os.path.exists(test_dir_t):
    os.makedirs(test_dir_t)

train_file_index = 0
val_file_index = 0
test_file_index = 0


# Loop over classes
for class_folder in os.listdir(input_dir):
    print('processing folder: '+class_folder)
    class_path = os.path.join(input_dir, class_folder)
    if os.path.isdir(class_path):
        # Get list of image files
        img_files = os.listdir(class_path)
        template_file = [f for f in img_files if f.lower().endswith('00.jpg')][0]
        # Remove template image
        img_files.remove(template_file)
        # Randomly select 30 images for training
        train_imgs_p = random.sample(img_files, num_train)
        # Randomly select 6 images for validation
        val_imgs_p = random.sample(list(set(img_files) - set(train_imgs_p)), num_val)
        # Use remaining 3 images for testing
        test_imgs_p = list(set(img_files) - set(train_imgs_p) - set(val_imgs_p))

        other_classes = [c for c in os.listdir(input_dir) if c != class_folder]
        img_files = []
        for folder in other_classes:
            img_list = os.listdir(os.path.join(input_dir, folder))
            img_list_w_path = [os.path.join(input_dir,folder, img) for img in img_list]
            img_files.extend(img_list_w_path)

        # Randomly select 30 images for training
        train_imgs_n = random.sample(img_files, num_train)
        # Randomly select 6 images for validation
        val_imgs_n = random.sample(list(set(img_files) - set(train_imgs_n)), num_val)
        # Use remaining 3 images for testing
        test_imgs_n = random.sample(list(set(img_files) - set(train_imgs_n) - set(val_imgs_n)), num_test)

        # Create pairs for training
        for img_file_p, img_file_n in zip(train_imgs_p, train_imgs_n):
            # Create a unique index for each pair
            train_file_index += 1
            copiedName = f'img{train_file_index:04d}.jpg'
            # Copy template file
            shutil.copy(os.path.join(class_path, template_file), os.path.join(train_dir_t, copiedName))
            # Copy positive sample
            shutil.copy(os.path.join(class_path, img_file_p), os.path.join(train_dir1, copiedName))
            # Copy negative sample
            shutil.copy(os.path.join(class_path, img_file_n), os.path.join(train_dir0, copiedName))

        # Create pairs for validation
        for img_file_p, img_file_n in zip(val_imgs_p, val_imgs_n):
            # Create a unique index for each pair
            val_file_index += 1
            copiedName = f'img{val_file_index:04d}.jpg'
            # Copy template file
            shutil.copy(os.path.join(class_path, template_file), os.path.join(val_dir_t, copiedName))
            # Copy positive sample
            shutil.copy(os.path.join(class_path, img_file_p), os.path.join(val_dir1, copiedName))
            # Copy negative sample
            shutil.copy(os.path.join(class_path, img_file_n), os.path.join(val_dir0, copiedName))

        # Create pairs for testing
        for img_file_p, img_file_n in zip(test_imgs_p, test_imgs_n):
            # Create a unique index for each pair
            test_file_index += 1
            copiedName = f'img{test_file_index:04d}.jpg'
            # Copy template file
            shutil.copy(os.path.join(class_path, template_file), os.path.join(test_dir_t, copiedName))
            # Copy positive sample
            shutil.copy(os.path.join(class_path, img_file_p), os.path.join(test_dir1, copiedName))
            # Copy negative sample
            shutil.copy(os.path.join(class_path, img_file_n), os.path.join(test_dir0, copiedName))
