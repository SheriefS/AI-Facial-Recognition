import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# import models
from MainModel import MainModel
from Variant1 import Variant1
from Variant2 import Variant2


def evaluate_model(model, val_loader, device, class_names):
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=range(len(class_names)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {type(model).__name__}')
    plt.show()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_precision = precision_score(all_labels, all_predictions, average='macro')
    macro_recall = recall_score(all_labels, all_predictions, average='macro')
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')

    micro_precision = precision_score(all_labels, all_predictions, average='micro')
    micro_recall = recall_score(all_labels, all_predictions, average='micro')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')

    return accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
val_loader = torch.load('validation.pth') 
class_names = ['Focused', 'happy', 'neutral', 'surprise'] 

models = [MainModel(), Variant1(), Variant2()]
metrics = []

for model in models:
    model_path = f'{type(model).__name__}.pth'
    model.load_state_dict(torch.load(model_path))

    results = evaluate_model(model, val_loader, device, class_names)
    metrics.append(results)

# Print out Metrics
for i, model in enumerate(['Main Model', 'Variant 1', 'Variant 2']):
    print(f"{model}: Accuracy: {metrics[i][0]:.2f}, Macro P: {metrics[i][1]:.2f}, Macro R: {metrics[i][2]:.2f}, Macro F: {metrics[i][3]:.2f}, Micro P: {metrics[i][4]:.2f}, Micro R: {metrics[i][5]:.2f}, Micro F: {metrics[i][6]:.2f}")
