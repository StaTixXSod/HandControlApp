import torch.nn as nn
import torch


class Generator(nn.Module):
    """
    Has to generate output with shape (batch, 63)
    random_size: latent size from the beginning
    """
    def __init__(self, random_size, output_size, num_classes, embed_size):
        super(Generator, self).__init__()
        self.random_size = random_size
        self.output_size = output_size

        self.gen = nn.Sequential(
            nn.Linear(random_size+embed_size, 64), nn.ReLU(inplace=True), nn.BatchNorm1d(64),
            nn.Linear(64, 128, bias=False), nn.ReLU(inplace=True), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 256, bias=False), nn.ReLU(inplace=True), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 512, bias=False), nn.ReLU(inplace=True), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, output_size)
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def forward(self, x, labels):
        embedding = self.embed(labels)
        x = torch.cat([x, embedding], dim=1)
        x = self.gen(x)
        return x


class Discriminator(nn.Module):
    """
    Has to distinguish real from fake
    INPUT shape: (batch, 63):
    """
    def __init__(self, input_size, num_classes):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_size+63, 128, bias=False), nn.ReLU(inplace=True), nn.BatchNorm1d(128),
            nn.Linear(128, 256, bias=False), nn.ReLU(inplace=True), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 32, bias=False), nn.ReLU(inplace=True), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.embed = nn.Embedding(num_classes, input_size)

    def forward(self, x, label):
        embedding = self.embed(label)
        x = torch.cat([x, embedding], dim=1)
        x = self.dis(x)
        return x
