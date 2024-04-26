import cv2
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Function to apply brightness adjustment and rotation augmentation
def adjust_brightness(image, brightness_factor_range=(0.5, 2.0)):
    brightness_factor = np.random.uniform(*brightness_factor_range)
    return cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

def rotate_image(image, rotation_angle_range=(-10, 10)):
    rows, cols = image.shape
    rotation_angle = np.random.uniform(*rotation_angle_range)
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))

# Function to load and preprocess dataset
def load_and_preprocess_dataset(dataset_path, img_size=(48, 48)):
    classes = ['Focused', 'happy', 'neutral', 'surprise']
    data = []
    labels = []

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        for image_name in images:
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Loading in grayscale

            if img_size is not None:
                image = cv2.resize(image, img_size)

            image = adjust_brightness(image)
            image = rotate_image(image)

            data.append(image)
            labels.append(class_name)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels)



    # Normalize pixel values to range [0, 1]
    data = data/255.0

    # Reshape data for compatibility with PyTorch (batch_size, channels, height, width)
    data = data.reshape(-1, 1, img_size[0], img_size[1])

    # Shuffle data and labels together
    data, labels = shuffle(data, labels, random_state=42)
    

    return data, labels, classes



def split_dataset(data, labels, test_size=0.15, val_size=0.1765, random_state=42):
    
    # First, split the data into a combined training+validation set, and a separate test set
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(data, labels, test_size=test_size, stratify=labels, random_state=random_state)

    # The size for validation split needs to be adjusted since it's calculated from the remaining training data
    adjusted_val_size = val_size / (1 - test_size)

    # Split the combined training+validation set into separate training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(train_val_data, train_val_labels, test_size=adjusted_val_size, stratify=train_val_labels, random_state=random_state)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels

#function for processing images that are used for testing the prediction in evaluate_model_2.py
def preprocess_image(image_path, img_size=(48, 48)):
    if not os.path.isfile(image_path):
        raise ValueError(f"File does not exist: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")
    
    image = cv2.resize(image, img_size)
    image = adjust_brightness(image)
    image = rotate_image(image)
    image = image / 255.0
    image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)  # Add batch and channel dimensions
    return torch.tensor(image, dtype=torch.float)

def plot_image_distribution(labels, class_names):
    # Count the number of occurrences for each class in the labels array
    from collections import Counter
    label_counts = Counter(labels)

    # Prepare data for plotting
    frequencies = [label_counts[class_name] for class_name in class_names]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, frequencies, color='skyblue')
    plt.xlabel('Class Names')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images Per Class')
    plt.show()