# =====================================
# === TRAINING CGAN ON CSV FILE  ======
# =====================================

import pandas as pd
from models.CGAN.CGAN import Generator, Discriminator
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from models.CGAN.utils import train_CGAN, plot_history, GestureDataset
import seaborn as sns

sns.set_style("darkgrid")

hot_labels = {
    "nothing": 0,
    "swiping": 1,
    "scrolling": 2,
    "pointing": 3
}

df = pd.read_csv("../../data/gestures.csv")
df['label'] = df['label'].map(hot_labels)

EPOCHS = 100
BATCH = 128
LATENT_SIZE = 32
EMBED_SIZE = 128
input_size = df.drop(['label'], axis=1).shape[1]  # 63
num_classes = df['label'].nunique()  # 4

dataset = GestureDataset(df)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

device = torch.device("cuda")

dis = Discriminator(input_size, num_classes).to(device)
gen = Generator(LATENT_SIZE, input_size, num_classes, EMBED_SIZE).to(device)

criterion = nn.MSELoss()
opt_D = torch.optim.Adam(dis.parameters(), lr=5e-4, amsgrad=True)
opt_G = torch.optim.Adam(gen.parameters(), lr=5e-4, amsgrad=True)

dis, gen, history = train_CGAN(dis, gen, opt_D, opt_G, criterion, loader, EPOCHS)
plot_history(history)
