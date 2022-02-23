import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda")


class GestureDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]


def train_dis(dis, gen, criterion, opt_D, X, y):
    batch = X.size(0)
    opt_D.zero_grad()
    real_ans = dis(X, y)
    real_loss = criterion(real_ans, torch.ones_like(real_ans, device=device))
    real_loss.backward()
    real_score = real_ans.mean().item()

    noise = torch.randn(batch, 32, device=device)
    fake_X = gen(noise, y)

    fake_ans = dis(fake_X, y)
    fake_loss = criterion(fake_ans, torch.zeros_like(fake_ans, device=device))
    fake_loss.backward()

    loss = real_loss + fake_loss
    opt_D.step()
    return loss.item(), real_score


def train_gen(dis, gen, criterion, opt_G, X, y):
    opt_G.zero_grad()
    batch = X.size(0)
    noise = torch.randn(batch, 32, device=device)
    fake_X = gen(noise, y)
    gen_ans = dis(fake_X, y)
    gen_loss = criterion(gen_ans, torch.ones_like(gen_ans))
    gen_loss.backward()

    fake_score = gen_ans.mean().item()
    opt_G.step()
    return gen_loss.item(), fake_score


def train_CGAN(dis, gen, opt_D, opt_G, criterion, loader, epochs):
    G_losses = []
    D_losses = []
    real_scores = []
    fake_scores = []

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            for X, y in loader:
                X = X.to(device).float()
                y = y.to(device)
                d_loss, real_score = train_dis(dis, gen, criterion, opt_D, X, y)
                g_loss, fake_score = train_gen(dis, gen, criterion, opt_G, X, y)

                G_losses.append(g_loss)
                D_losses.append(d_loss)
                real_scores.append(fake_score)
                fake_scores.append(fake_score)

            pbar.update(1)
            pbar.set_postfix(d_loss=d_loss, g_loss=g_loss)

    torch.save(gen.state_dict(), "Conditional_GEN.pth")
    return dis, gen, (G_losses, D_losses, real_scores, fake_scores)


def plot_history(history):
    G_losses, D_losses, real_scores, fake_scores = history
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle('CONDITIONAL GAN LEARNING PROCESS', fontsize=20)
    ax[0].set_title("Losses")
    ax[0].plot(G_losses, label="G_losses")
    ax[0].plot(D_losses, label="D_losses")
    ax[0].legend()
    ax[1].set_title("Scores")
    ax[1].plot(real_scores, label="real_scores")
    ax[1].plot(fake_scores, label="fake_scores")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("../models/GANS/Conditional_GAN_learning_process.png")
    print("Pic was saved")
    plt.show()
