# ===================================================
# === TRAINING HAND POSE DETECTOR ON CSV FILE  ======
# ===================================================

import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import pandas as pd
from utils import train_model, plot_history, GestureDataset
from models.HPD.HandPoseDetector import HandPoseModel
import seaborn as sns
sns.set_style("darkgrid")


INPUT_FEATURES = 63
NUM_CLASSES = 4
BATCH_SIZE = 128
EPOCHS = 20
data_path = "/data/gestures.csv"
gen_data_path = "/generated"

labels = {
    "nothing": 0,
    "swiping": 1,
    "scrolling": 2,
    "pointing": 3
}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_gesture_dataloader(data_path, gen_path, use_gen_data=False):
    """
    Returns TRAIN and TEST DataLoader,
    that contains coordinates as data
    and gestures as labels
    :param data_path: path to original dataset
    :param gen_path: path to folder, that contains generative data
    :param use_gen_data: True for using generated data
    :return: Train dataloader, Test dataloader
    """
    datasets = []

    df = pd.read_csv(data_path)
    df['label'] = df['label'].map(labels)
    dataset = GestureDataset(df)
    datasets.append(dataset)

    if use_gen_data:
        gen_gestures = os.listdir(gen_path)
        for gest in gen_gestures:
            gen_df = pd.read_csv(os.path.join(gen_path, gest))
            gen_df['label'] = gen_df['label'].map(labels)
            datasets.append(GestureDataset(gen_df))

    ds = ConcatDataset(datasets=datasets)

    train_size = int(0.8 * len(ds))
    valid_size = int(0.2 * len(ds))+1
    train_ds, valid_ds = random_split(ds, [train_size, valid_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader


train_loader, valid_loader = get_gesture_dataloader(data_path, gen_data_path, use_gen_data=True)

HandPoseModel = HandPoseModel(INPUT_FEATURES, NUM_CLASSES).to(device)

optimizer = torch.optim.Adam(HandPoseModel.parameters(), lr=3e-4, amsgrad=True)
criterion = torch.nn.CrossEntropyLoss()

model, history = train_model(
    model=HandPoseModel,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=EPOCHS,
    name="HandPoseDetector")

plot_history(history)
