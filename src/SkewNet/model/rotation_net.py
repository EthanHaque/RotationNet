import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
import torchvision.transforms as transforms


class RotationNetMobileNetV3Backbone(nn.Module):
    def __init__(self):
        super(RotationNetMobileNetV3Backbone, self).__init__()
        self.base_model = mobilenet_v3_large(weights="DEFAULT", width_mult=1.0, reduced_tail=False, dilated=False)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1000, 1)


    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


    def train_transform(self, x):
        image_transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=False),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ])
        return image_transform(x)


    def evaluation_transform(self, x):
        image_transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=False),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_transform(x)
    

class RotationNetSmallNetworkTest(nn.Module):
    def __init__(self):
        super(RotationNetSmallNetworkTest, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = self.global_avg_pool(x)
        x = x.view(-1, 64)
        x = self.fc(x)

        return x


    def train_transform(self, x):
        image_transform = transforms.Compose([
            transforms.Resize((1200, 900), antialias=False),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])
        return image_transform(x)


    def evaluation_transform(self, x):
        image_transform = transforms.Compose([
            transforms.Resize((1200, 900), antialias=False),
        ])
        return image_transform(x)
    

class RotationNetLargeNetworkTest(nn.Module):
    def __init__(self):
        super(RotationNetLargeNetworkTest, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        x = self.global_avg_pool(x)
        x = x.view(-1, 128)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


    def train_transform(self, x):
        image_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        ])
        return image_transform(x)


    def evaluation_transform(self, x):
        image_transform = transforms.Compose([])
        return image_transform(x)
    