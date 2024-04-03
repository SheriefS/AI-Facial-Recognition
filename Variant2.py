import torch.nn as nn
import torch.nn.functional as F

num_classes = 4
# Define the CNN architecture
class Variant2(nn.Module):
    def __init__(self):
        super(Variant2, self).__init__()
        # Convolutional layers with Batch Normalization
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.layer4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # self.layer5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply convolutions, batch normalization, activation function (ReLU), and max pooling
        x = self.pool(F.leaky_relu(self.bn1(self.layer1(x))))
        #print("After layer1:", x.shape)
        x = self.pool(F.leaky_relu(self.bn2(self.layer2(x))))
        #print("After layer1:", x.shape)
        x = self.pool(F.leaky_relu(self.bn3(self.layer3(x))))
        #print("After layer1:", x.shape)
        x = self.pool(F.leaky_relu(self.bn4(self.layer4(x))))
        #print("After layer1:", x.shape)
        # x = self.pool(F.leaky_relu(self.bn5(self.layer5(x))))
        #print("After layer1:", x.shape)

        # Flatten the output for the fully connected layers
        x = x.view(-1, 256 * 3 * 3)

        # Apply fully connected layers with ReLU, dropout, and output layer
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x