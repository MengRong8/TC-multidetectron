import torch
import torch.nn as nn

# class ImageLinearModel(nn.Module):
#     def __init__(self):
#         super(ImageLinearModel, self).__init__()
#         self.linear = nn.Linear(1408, 1)

#     def forward(self, image_features):
#         x = self.linear(image_features)
#         return x

class SingleClassCBMEncoder(nn.Module):
    """Single binary classifier for one object class (CBM per-class encoder)"""
    def __init__(self, input_dim=1408):
        super(SingleClassCBMEncoder, self).__init__()
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, image_features):
        """
        Args:
            image_features: [batch_size, 1408] BLIP2 image features
        Returns:
            [batch_size, 1] binary logits (real=0, fake=1)
        """
        return self.classifier(image_features)
    
class ObjectClassCBMEncoder(nn.Module):
    def __init__(self):
        super(ObjectClassCBMEncoder, self).__init__()
        self.classifiers = nn.ModuleList([
            nn.Linear(1408, 1)  # 移除 Sigmoid，輸出 logits
            for _ in range(300)
        ])

    def forward(self, image_features, classifier_index):
        """
        Args:
            image_features: [batch_size, 1408]
            classifier_index: int, 指定使用哪個 classifier (0-299)
        Returns:
            [batch_size, 1] logits（未經 sigmoid）
        """
        return self.classifiers[classifier_index](image_features)

class ObjectClassCBMPredictor(nn.Module):
    def __init__(self):
        super(ObjectClassCBMPredictor, self).__init__()
        self.linear = nn.Linear(300, 1)

    def forward(self, logits_per_image):
        x = self.linear(logits_per_image)
        return x
    
class MiRAGeImg(nn.Module):
    def __init__(self):
        super(MiRAGeImg, self).__init__()
        self.linear = nn.Linear(301, 1)
        # 移除 self.sigmoid

    def forward(self, logits_per_image):
        x = self.linear(logits_per_image)
        # 輸出 logits
        return x


class SingleClassCBMMLPEncoder(nn.Module):
    def __init__(self, input_dim=1408, hidden_dim=256, dropout=0.2):
        super(SingleClassCBMMLPEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image_features):
        return self.mlp(image_features)


class ObjectClassCBMMLPEncoder(nn.Module):
    def __init__(self, num_concepts=300, input_dim=1408, hidden_dim=256, dropout=0.2):
        super(ObjectClassCBMMLPEncoder, self).__init__()
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_concepts)
        ])

    def forward(self, image_features, classifier_index):
        """
        Args:
            image_features: [batch_size, 1408]
            classifier_index: int, 使用哪個classifier (0-299)
        Returns:
            [batch_size, 1] logits
        """
        return self.classifiers[classifier_index](image_features)