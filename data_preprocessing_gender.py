# data_preprocessing_gender.py
import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to apply brightness adjustment and rotation augmentation
def adjust_brightness(image, brightness_factor_range=(0.5, 2.0)):
    brightness_factor = np.random.uniform(*brightness_factor_range)
    return cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

def rotate_image(image, rotation_angle_range=(-10, 10)):
    rows, cols = image.shape
    rotation_angle = np.random.uniform(*rotation_angle_range)
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))

# Define load_and_preprocess_dataset_gender to reflect folder structure of data split by gender
def load_and_preprocess_dataset_gender(dataset_path, img_size=(48, 48)):
    genders = ['male', 'female', 'non-binary']
    emotions = ['Focused', 'happy', 'neutral', 'surprise']
    data = {}
    labels = {}

    for gender in genders:
        data[gender] = []
        labels[gender] = []

        for emotion in emotions:
            emotion_gender_path = os.path.join(dataset_path, gender, emotion)
            if not os.path.isdir(emotion_gender_path):
                continue

            images = os.listdir(emotion_gender_path)
            for image_name in images:
                image_path = os.path.join(emotion_gender_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    if img_size is not None:
                        image = cv2.resize(image, img_size)
                    image = adjust_brightness(image)
                    image = rotate_image(image)
                    data[gender].append(image)
                    labels[gender].append(emotion)  # Just store emotion as label for simplicity

        # Normalize data
        data[gender] = np.array(data[gender], dtype=np.float32) / 255.0
        data[gender] = data[gender].reshape(-1, 1, img_size[0], img_size[1])

    return data, labels, emotions  # No need for label encoder if labels are already correct

def prepare_data_for_gender(data, labels, gender, test_size=0.15, val_size=0.15):
    # Initialize Label Encoder
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)  # Convert string labels to integer

    print(f"Preparing data for gender: {gender} - Total samples: {len(data)}")
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        data, encoded_labels, test_size=test_size, stratify=encoded_labels, random_state=42)

    print(f"Test set size for {gender}: {len(test_data)}")
    adjusted_val_size = val_size / (1 - test_size)
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=adjusted_val_size, stratify=train_val_labels, random_state=42)
    
    print(f"Training set size for {gender}: {len(train_data)}, Validation set size: {len(val_data)}")
        # Create a bar plot for the number of images per class within the gender
    plt.figure(figsize=(12, 6))
    unique, counts = np.unique(np.concatenate((train_labels, val_labels, test_labels)), return_counts=True)
    sns.barplot(x=le.inverse_transform(unique), y=counts)
    plt.title(f'Number of Images per Class for {gender.capitalize()} Gender')
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'images_per_class_{gender}.png')

    return train_data, val_data, test_data, train_labels, val_labels, test_labels
