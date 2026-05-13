import torch
from torchvision.models import resnet18
from torch import nn


def get_3channel_resnet18():
    """Pseudo 3-channel ResNet18: duplicates ch0 to form [img, img, mask]."""
    model = resnet18(pretrained=True)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    class CustomResNet18(nn.Module):
        def __init__(self, feature_extractor, classifier):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.classifier = classifier

        def forward(self, x):
            x = torch.cat([x[:, 0:1], x[:, 0:1], x[:, 1:2]], dim=1)
            features = torch.flatten(self.feature_extractor(x), start_dim=1)
            return features, self.classifier(features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return CustomResNet18(feature_extractor, model.fc).to(device)
