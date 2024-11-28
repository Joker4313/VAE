import torch
from torch import nn
import torch.nn.functional as F


# Input img -> Hidden dim -> mead, std -> Parameterization trick -> Decoder -> Output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_dim, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img_2hid = nn.Linear(in_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, in_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theto(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_parametrized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_parametrized)
        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)
    vae = VariationalAutoEncoder(in_dim=28 * 28)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
