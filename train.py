import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import load_and_preprocess_dataset, split_dataset
from MainModel import MainModel
from Variant1 import Variant1
from Variant2 import Variant2

"""
Functions to save and load the dataset
This is done to ensure that the data is only loaded and preprocessed once to ensure that there
is consistent data being used across all models. This prevents the evaluation of an unchanged
model from changing if another model uses the data to train. This way the influence of the 
actual model can be more accurately assessed.
"""

def save_preprocessed_data(data, labels, class_names, file_path):
    torch.save({'data': data, 'labels': labels, 'class_names': class_names}, file_path)

def load_preprocessed_data(file_path):
    loaded_data = torch.load(file_path)
    return loaded_data['data'], loaded_data['labels'], loaded_data['class_names']

# Define paths for saving/loading preprocessed data
preprocessed_data_path = 'preprocessed_data.pth'

# Check if the preprocessed data exists
if os.path.exists(preprocessed_data_path):
    # Load preprocessed data
    data, labels, class_names = load_preprocessed_data(preprocessed_data_path)
else:
    # Preprocess data
    dataset_path = "images_AI_dataset/images/train"
    max_samples_per_class = 800
    data, labels, class_names = load_and_preprocess_dataset(dataset_path, max_samples_per_class)
    
    # Save preprocessed data for future use
    save_preprocessed_data(data, labels, class_names, preprocessed_data_path)

train_data, val_data, test_data, train_labels, val_labels, test_labels = split_dataset(data, labels)

# # Reshape data for compatibility with PyTorch
# train_data = train_data.reshape(-1, 1, 48, 48)
# test_data = test_data.reshape(-1, 1, 48, 48)
# val_data = val_data.reshape(-1, 1, 48, 48)

label_encoder = LabelEncoder()

train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)
val_labels = label_encoder.transform(val_labels)

# Convert numpy arrays to PyTorch tensors
train_data = torch.tensor(train_data).float()
train_labels = torch.tensor(train_labels).long()
val_data = torch.tensor(val_data).float()
val_labels = torch.tensor(val_labels).long()
test_data = torch.tensor(test_data).float()
test_labels = torch.tensor(test_labels).long()

# Create TensorDataset wrappers
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def train_model(model, num_epochs, model_save_path):
    print("Starting training...")
    
# Specify the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #patience value is set to stop the loop if overfitting is detected
    #3 consecutive epoch without a decrease in loss will halt the loop
    patience = 3
    best_val_loss = np.inf
    counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at Epoch {epoch+1} with Validation Loss: {val_loss:.4f}")
        else:
            counter += 1

        if counter >= patience:
            print("Validation loss has not improved for {} epochs. Early stopping...".format(patience))
            break
    torch.save(val_loader, 'validation.pth')

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('model_variant', type=int, help='Model variant to train: 1 for MainModel, 2 for Variant1, 3 for Variant2')
    # Dictionary mapping command-line arguments to model classes

    args = parser.parse_args()

    models = { 1: MainModel, 2: Variant1, 3: Variant2 }

    # Choose the model based on the argument provided
    if args.model_variant in models:
        model_class = models[args.model_variant]
        model = model_class()  # Initialize the model
    else:
        raise ValueError("Invalid model variant selected. Choose 1 for MainModel, 2 for Variant1, or 3 for Variant2.")


    num_epochs = 20
    model_save_path = f"{model_class.__name__}.pth" 
    train_model(model, num_epochs, model_save_path)