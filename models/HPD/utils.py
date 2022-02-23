import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class GestureDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]


def train_epoch(model, optimizer, criterion, loader):
    model.train()
    full_loss = 0
    full_f_score = 0
    # for data, labels in tqdm(loader, leave=False):
    for data, labels in loader:
        optimizer.zero_grad()
        data = data.to(device).float()
        labels = labels.to(device)

        out = model(data)

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        full_loss += loss.item()

        y_pred = torch.argmax(torch.softmax(out, dim=1), dim=-1)
        f_score = f1_score(labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
        full_f_score += f_score

    full_loss = full_loss / len(loader)
    full_f_score = full_f_score / len(loader)
    return full_loss, full_f_score


def eval_epoch(model, criterion, loader):
    model.eval()
    full_loss = 0
    full_f_score = 0
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device).float()
            labels = labels.to(device)

            out = model(data)
            loss = criterion(out, labels)
            full_loss += loss.item()

            y_pred = torch.argmax(torch.softmax(out, dim=1), dim=-1)
            f_score = f1_score(labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='weighted')
            full_f_score += f_score

    full_loss = full_loss / len(loader)
    full_f_score = full_f_score / len(loader)
    return full_loss, full_f_score


def train_model(model, optimizer, criterion, train_loader, valid_loader, epochs, name: str):
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_f_score": [],
        "valid_f_score": []
    }

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            train_loss, train_score = train_epoch(model, optimizer, criterion, train_loader)
            valid_loss, valid_score = eval_epoch(model, criterion, valid_loader)

            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_f_score'].append(train_score)
            history['valid_f_score'].append(valid_score)

            pbar.update(1)
            pbar.set_postfix(valid_loss=valid_loss, valid_score=valid_score)

    torch.save(model.state_dict(), f"/{name}.pth")
    return model, history


def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    ax[0].set_title("Losses")
    ax[0].plot(history['train_loss'], label="Train loss")
    ax[0].plot(history['valid_loss'], label="Valid loss")
    ax[0].legend()
    ax[1].set_title("Scores")
    ax[1].plot(history['train_f_score'], label="Train Score")
    ax[1].plot(history['valid_f_score'], label="Valid Score")
    ax[1].legend()
    plt.tight_layout()

    plt.savefig("../../models/HPD/HPD_train_process.png")
    plt.show()
