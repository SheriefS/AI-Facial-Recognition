import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_preprocessing_gender import load_and_preprocess_dataset_gender, prepare_data_for_gender
from MainModel import MainModel
from sklearn.preprocessing import LabelEncoder
import os
import seaborn as sns
import matplotlib.pyplot as plt



def load_model(model_path, device):
    model = MainModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def save_data_splits(data_splits_path, train_data, val_data, test_data, train_labels, val_labels, test_labels):
    torch.save({
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels
    }, data_splits_path)

def load_data_splits(data_splits_path):
    splits = torch.load(data_splits_path)
    return (splits['train_data'], splits['val_data'], splits['test_data'],
            splits['train_labels'], splits['val_labels'], splits['test_labels'])

def evaluate_model_by_gender(model, dataset_path, device):
    data, labels, class_names = load_and_preprocess_dataset_gender(dataset_path)
    results = {}

    for gender in data.keys():
        data_splits_path = f"{gender}_data_splits.pth"

        if os.path.exists(data_splits_path):
            train_data, val_data, test_data, train_labels, val_labels, test_labels = load_data_splits(data_splits_path)
            print(f"Loaded existing data splits for {gender}.")
        else:
            # If the split file does not exist, do the split and save it
            train_data, val_data, test_data, train_labels, val_labels, test_labels = prepare_data_for_gender(data[gender], labels[gender], gender)
            save_data_splits(data_splits_path, train_data, val_data, test_data, train_labels, val_labels, test_labels)
            print(f"Created and saved data splits for {gender}.")

        # Convert to tensors and create DataLoader
        test_tensor_data = torch.tensor(test_data, dtype=torch.float32)
        test_tensor_labels = torch.tensor(test_labels, dtype=torch.long)
        test_dataset = TensorDataset(test_tensor_data, test_tensor_labels)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device, class_names)
        results[gender] = {
            'Accuracy': f"{accuracy:.2f}",
            'Precision': f"{precision:.2f}",
            'Recall': f"{recall:.2f}",
            'F1-Score': f"{f1:.2f}"
        }

    return results

def evaluate_model(model, data_loader, device, classes):
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    


    # Compute metrics with zero_division parameter set to 1
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'MainModel.pth'
    model = load_model(model_path, device)
    dataset_path = "images_AI_dataset_gender/images/train"
    
    results = evaluate_model_by_gender(model, dataset_path, device)

    # Compute averages
    avg_accuracy = np.mean([float(results[gender]['Accuracy']) for gender in results])
    avg_precision = np.mean([float(results[gender]['Precision']) for gender in results])
    avg_recall = np.mean([float(results[gender]['Recall']) for gender in results])
    avg_f1_score = np.mean([float(results[gender]['F1-Score']) for gender in results])

    # Print formatted results
    print("Performance by Gender:")
    for gender, metrics in results.items():
        print(f"{gender.capitalize()}: {metrics}")
    
    print("\nAverage Performance Across All Genders:")
    print(f"Average Accuracy: {avg_accuracy:.2f}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1-Score: {avg_f1_score:.2f}")
