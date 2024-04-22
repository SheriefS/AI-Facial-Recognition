import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from data_preprocessing import load_and_preprocess_dataset
from MainModel import MainModel
import numpy as np

def train_model(data, labels, train_idx, device):
    # Initialize the model and send it to the device (GPU or CPU)
    model = MainModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert training data and labels into tensors and send them to the device
    train_data = torch.tensor(data[train_idx], dtype=torch.float).to(device)
    train_labels = torch.tensor(labels[train_idx], dtype=torch.long).to(device)
    
    # Create a DataLoader for the training data
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=64, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(20):  # Number of epochs
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # Save the model state
    torch.save(model.state_dict(), f'model_fold_{fold+1}.pth')
    
    # Save the indices for the fold
    np.save(f'test_indices_fold_{fold+1}.npy', test_idx)

    return model


if __name__ == "__main__":
    dataset_path = "images_AI_dataset/images/train"
    data, labels, _ = load_and_preprocess_dataset(dataset_path)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert labels to a numeric format
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Iterate over each fold for training and save the models
    for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
        model = train_model(data, encoded_labels, train_idx, device)
        
        # Confirm fold save and training
        print(f"Saved model and indices for fold {fold+1}")