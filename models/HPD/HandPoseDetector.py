import torch.nn as nn


class HandPoseModel(nn.Module):
    def __init__(self, input_features, num_classes):
        super(HandPoseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(128, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(256, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(128, num_classes, bias=False)
        )

    def forward(self, x):
        return self.model(x)
