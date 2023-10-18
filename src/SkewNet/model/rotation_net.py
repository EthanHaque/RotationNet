import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large


class RotationNet(nn.Module):
    def __init__(self):
        super(RotationNet, self).__init__()
        self.base_model = mobilenet_v3_large(weights="DEFAULT", width_mult=1.0, reduced_tail=False, dilated=False)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


if __name__ == '__main__':
    model = RotationNet()
    print(model)
    in_tensor = torch.randn(1, 3, 224, 224)
    out_tensor = model(in_tensor)
    print(out_tensor.shape)