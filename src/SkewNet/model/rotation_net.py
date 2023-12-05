import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, resnet50, convnext_tiny
import torchvision.transforms as transforms


class ModelRegistry:
    registry = {}

    @classmethod
    def register(cls, name):
        def inner_wrapper(wrapped_class):
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_model(cls, name):
        if name in cls.registry:
            return cls.registry[name]()
        else:
            raise ValueError(f"Model {name} not found in registry")

    @classmethod
    def get_model_names(cls):
        return list(cls.registry.keys())

    def __dict__(self):
        return self.registry

    def __len__(self):
        return len(self.registry)

    def __getitem__(self, key):
        return self.registry[key]

    def __contains__(self, key):
        return key in self.registry

    def __iter__(self):
        return iter(self.registry)

    def __repr__(self):
        return repr(self.registry)

    def __str__(self):
        return str(self.registry)


################## Transfer Learning Test Networks ##################


@ModelRegistry.register("ConvNeXtTinyBackbone")
class RotationNetConvNeXtTinyBackbone(nn.Module):
    def __init__(self):
        super(RotationNetConvNeXtTinyBackbone, self).__init__()
        self.base_model = convnext_tiny(weights="DEFAULT")
        self.base_model.classifier = nn.Identity()
        self.fc1 = nn.Linear(768, 1)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


@ModelRegistry.register("MobileNetV3Backbone")
class RotationNetMobileNetV3Backbone(nn.Module):
    def __init__(self):
        super(RotationNetMobileNetV3Backbone, self).__init__()
        self.base_model = mobilenet_v3_large(weights="DEFAULT", width_mult=1.0, reduced_tail=False, dilated=False)
        self.base_model.classifier = nn.Identity()
        self.fc1 = nn.Linear(960, 1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)

        return x


@ModelRegistry.register("Resnet50Backbone")
class RotationNetResnet50Backbone(nn.Module):
    def __init__(self):
        super(RotationNetResnet50Backbone, self).__init__()
        self.base_model = resnet50(weights="DEFAULT")
        self.fc1 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)

        return x


#################### Test Networks ####################


@ModelRegistry.register("SmallTestNetwork")
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


@ModelRegistry.register("LargeTestNetwork")
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


@ModelRegistry.register("HugeTestNetwork")
class RotationNetHugeNetworkTest(nn.Module):
    def __init__(self):
        super(RotationNetHugeNetworkTest, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.pool(self.relu(self.bn5(self.conv5(x))))
        x = self.pool(self.relu(self.bn6(self.conv6(x))))
        x = self.pool(self.relu(self.bn7(self.conv7(x))))
        x = self.pool(self.relu(self.bn8(self.conv8(x))))

        x = self.global_avg_pool(x)
        x = x.view(-1, 512)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
