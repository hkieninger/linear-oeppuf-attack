import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from puf import Puf

class FFNNClone(Puf):
    """
    tries to learn a PUF from CRP observations using a simple feed forward neural network
    """

    def __init__(self, challenges : npt.NDArray[np.floating], responses : npt.NDArray[np.floating], hidden_layers : list[int] =[128, 64], learning_rate : float = 1e-3, epochs : int = 10, batch_size=16, plot=False):
        crp_count, challenge_length = challenges.shape
        _, response_length = responses.shape

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.from_numpy(challenges.astype(np.float32)).to(self.device)
        Y = torch.from_numpy(responses.astype(np.float32)).to(self.device)

        self.nn = FeedForwardNN(challenge_length, hidden_layers, response_length).to(self.device)
        opt = optim.Adam(self.nn.parameters(), lr=learning_rate)
        loss_fn = lambda pred, target: torch.mean(torch.norm(pred - target, dim=1)) # loss_fn = nn.MSELoss()

        self.nn.train()
        losses = []
        for epoch in range(epochs):
            perm = torch.randperm(crp_count, device=self.device)
            Xp = X[perm]; Yp = Y[perm]
            for i in range(0, crp_count, batch_size):
                xb = Xp[i:i+batch_size]; yb = Yp[i:i+batch_size]
                opt.zero_grad()
                pred = self.nn(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        if plot:
            self.train_losses = losses
            plt.figure()
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.yscale('log')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.grid(True)
            plt.show()

        self.nn.eval()

    def evaluate(self, challenges : npt.NDArray[np.floating]):
        if challenges.ndim == 1:
            inp = torch.from_numpy(challenges.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.nn(inp).cpu().numpy().squeeze(0)
            return out
        else:
            inp = torch.from_numpy(challenges.astype(np.float32)).to(self.device)
            with torch.no_grad():
                out = self.nn(inp).cpu().numpy()
            return out

class FeedForwardNN(nn.Module):

    def __init__(self, in_dim, h_dims, out_dim):
        super().__init__()
        layers = []
        for hidden_dim in h_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Sigmoid())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)