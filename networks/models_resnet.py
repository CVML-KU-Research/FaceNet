import torch
import torch.nn as nn
from torchvision import models
from facenet_pytorch import InceptionResnetV1


class FaceNetModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceNetModel, self).__init__()
        # Load pretrained ResNet50
        self.base = models.resnet50(pretrained=True)
        
        # Gradually unfreeze layers
        # layers_to_unfreeze = ['layer4', 'layer3']
        layers_to_unfreeze = ['layer4']

        for name, param in self.base.named_parameters():
            param.requires_grad = any(layer in name for layer in layers_to_unfreeze)
            
        # Improved feature extraction
        in_features = self.base.fc.in_features
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Advanced classifier with dropouts and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    
class ImprovedFaceNet(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedFaceNet, self).__init__()
        self.base = InceptionResnetV1(pretrained='vggface2')
        
        # Unfreeze later layers
        for param in self.base.parameters():
            param.requires_grad = False
        for block in [self.base.block8]:
        # for block in [self.base.block8, self.base.mixed_7a, self.base.repeat_2]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Improved classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.base(x)
        return self.classifier(features)

    
class ArcFaceModel(nn.Module):
    def __init__(self, num_classes):
        super(ArcFaceModel, self).__init__()
        self.base = models.resnet50(pretrained=True)
        
        # Selective layer unfreezing
        # layers_to_unfreeze = ['layer4', 'layer3']
        layers_to_unfreeze = ['layer4']

        for name, param in self.base.named_parameters():
            param.requires_grad = any(layer in name for layer in layers_to_unfreeze)
        
        # Channel attention module
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, 1),
            nn.Sigmoid()
        )
        
        # Improved classifier
        in_features = self.base.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        
        # Apply attention
        att = self.attention(x)
        x = x * att
        
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    
class VGGFaceModel(nn.Module):
    def __init__(self, num_classes):
        super(VGGFaceModel, self).__init__()
        self.base = models.vgg16(pretrained=True)
        
        # Selective layer unfreezing
        for param in self.base.parameters():
            param.requires_grad = False
        
        # Unfreeze later layers
        for layer in self.base.features[24:]:
            for param in layer.parameters():
                param.requires_grad = True
                
        # Feature pooling with attention
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Improved classifier
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(512 * 3 * 3, 4096),  # Adjusted for VGG16 output size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base.features(x)
        # print(f"Shape after feature extraction: {x.shape}")
        # Apply attention
        att = self.attention(x)
        x = x * att
        # print(f"Shape after attention: {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"Shape after flattening: {x.shape}")
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Example usage
    # model = FaceNetModel(num_classes=34)
    model = VGGFaceModel(num_classes=34)  # Example with 34 classes
    # print(model)

    # Test with a random input
    x = torch.randn(4, 3, 112, 112)  # Batch size of 4, 3 channels, 112x112 image
    output = model(x)
    print(output.shape)  # Should be [4, 34] for num_classes=34