import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_and_preprocess_dataset
from MainModel import MainModel

def evaluate_model(model, data, labels, test_idx, device):
    # Create a DataLoader for the test data
    test_data = torch.tensor(data[test_idx], dtype=torch.float).to(device)
    test_labels = torch.tensor(labels[test_idx], dtype=torch.long).to(device)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=64, shuffle=False)

    # Evaluate the model
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate and return metrics
    return {
        
        'macro_precision': precision_score(all_labels, all_predictions, average='macro'),
        'macro_recall': recall_score(all_labels, all_predictions, average='macro'),
        'macro_f1_score': f1_score(all_labels, all_predictions, average='macro'),
        'micro_precision': precision_score(all_labels, all_predictions, average='micro'),
        'micro_recall': recall_score(all_labels, all_predictions, average='micro'),
        'micro_f1_score': f1_score(all_labels, all_predictions, average='micro'),
        'accuracy': accuracy_score(all_labels, all_predictions)
    }

if __name__ == "__main__":
    dataset_path = "images_AI_dataset/images/train"
    data, labels, _ = load_and_preprocess_dataset(dataset_path, 800)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert labels to a numeric format
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    results = []
    # Iterate over each fold for evaluation
    for fold in range(1, kf.get_n_splits() + 1):
        
        # Load the saved model
        model = MainModel().to(device)
        model.load_state_dict(torch.load(f'model_fold_{fold}.pth'))

        # Load the test indices
        test_idx = np.load(f'test_indices_fold_{fold}.npy')

        # Evaluate the model
        metrics = evaluate_model(model, data, encoded_labels, test_idx, device)
        results.append(metrics)
        formatted_metrics = {k: f'{v:.2f}' for k, v in metrics.items()}
        print(f"Fold {fold+1}: {formatted_metrics}")

    # Calculate and print the average of the metrics
    avg_results = {metric: np.mean([result[metric] for result in results]) for metric in results[0]}
    formatted_avg_results = {k: f'{v:.2f}' for k, v in avg_results.items()}
    print(f"Average metrics: {formatted_avg_results}")
