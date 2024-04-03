import torch.nn as nn
import torch.nn.functional as F

num_classes = 4
# Define the CNN architecture
class Variant1(nn.Module):
    def __init__(self):
        super(Variant1, self).__init__()
        # Define convolutional layers
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #Output channel to 1 for grayscale images
        self.bn1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Define max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(0.5)

        # Define fully connected layers
        #Training data images for 48x48 pixels
        self.fc1 = nn.Linear(128 * 6 * 6, 512)  # Adjust the input size based on your image dimensions
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)  # num_classes to be defined based on your dataset

    def forward(self, x):
        # Apply convolutions, activation function (ReLU), and max pooling
        x = self.pool(F.leaky_relu(self.layer1(x)))
        #print("After layer1:", x.shape)
        x = self.pool(F.leaky_relu(self.layer2(x)))
        #print("After layer2:", x.shape)
        x = self.pool(F.leaky_relu(self.layer3(x)))
        #print("After layer3:", x.shape)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 128 * 6* 6)  # Adjusted based on 48x48 pixel resolution

        # Apply fully connected layers with ReLU and output layer
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)  # No activation function here, assuming using CrossEntropyLoss which includes Softmax

        return x