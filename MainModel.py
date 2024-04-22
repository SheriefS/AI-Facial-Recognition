import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

num_classes = 4
# Define a new CNN architecture
class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        # Convolutional layers with Batch Normalization
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply convolutions, batch normalization, activation function (ReLU), and max pooling
        x = self.pool(F.relu(self.bn1(self.layer1(x))))
        #print("After layer1:", x.shape)
        x = self.pool(F.relu(self.bn2(self.layer2(x))))
        #print("After layer2:", x.shape)
        x = self.pool(F.relu(self.bn3(self.layer3(x))))
        #print("After layer3:", x.shape)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 128 * 7 * 7)

        # Apply fully connected layers with ReLU, dropout, and output layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x