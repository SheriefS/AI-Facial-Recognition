import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import preprocess_image

# Import your model architecture
from MainModel import MainModel  # Adjust as necessary for your model

index_to_class = {0: 'Focused', 1: 'happy', 2: 'neutral', 3: 'surprise'}

def load_model(model_path, device):
    model = MainModel().to(device)  # Initialize your model
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the saved model
    model.eval()
    return model

def evaluate_on_dataset(model, data_loader, device):
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    #accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Accuracy on dataset: {accuracy:.2f}')
    macro_precision = precision_score(all_labels, all_predictions, average='macro')
    print(f'Macro precision on dataset: {macro_precision:.2f}')
    macro_recall = recall_score(all_labels, all_predictions, average='macro')
    print(f'Macro Recall on dataset: {macro_recall:.2f}')
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    print(f'Macro F1 score on dataset: {macro_f1:.2f}')

    micro_precision = precision_score(all_labels, all_predictions, average='micro')
    print(f'Micro precision on dataset: {micro_precision:.2f}')
    micro_recall = recall_score(all_labels, all_predictions, average='micro')
    print(f'Micro Recall on dataset: {micro_recall:.2f}')
    micro_f1 = f1_score(all_labels, all_predictions, average='micro')
    print(f'Micro F1 score on dataset: {micro_f1:.2f}')


def predict_images_in_folder(model, folder_path, device):
    
    
    # Check for image files in the specified directory
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in the specified directory.")
        return
    
    for image_file in image_files:
        full_image_path = os.path.join(folder_path, image_file)  # Combine the folder path with the image filename
        #print(f"Processing {full_image_path}...")  # Optional: Print the full path being processed
        
        try:
            # Ensure preprocess_image function is prepared to handle the full path to an image
            image_tensor = preprocess_image(full_image_path).to(device)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue  # Skip this file and continue with the next
        
        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        class_name = index_to_class[prediction]
        # Modify this to print class names if available
        print(f"The prediction for filename: {image_file} is {class_name}")

### Main Script Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'MainModel.pth'  # Ensure this is the correct path to your saved model
    model = load_model(model_path, device)  # Ensure load_model function properly loads the model

     #Assuming val_loader is defined similar to how it's loaded in your training script
    val_loader = torch.load('validation.pth')
    evaluate_on_dataset(model, val_loader, device)
    
    folder_path = './single_image'  # Adjust to your folder path
    predict_images_in_folder(model, folder_path, device)