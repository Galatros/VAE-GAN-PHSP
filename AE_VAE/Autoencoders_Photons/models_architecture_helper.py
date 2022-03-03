import torch
import torch.nn as nn


class VAE_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 5),
            nn.ReLU(),
            nn.BatchNorm1d(5),
            torch.nn.Linear(5, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )

        self.z_mean = nn.Sequential(
            nn.Linear(4, 3)
        )

        self.z_log_var = nn.Sequential(
            nn.Linear(4, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.BatchNorm1d(5),
            nn.Linear(5, 6)
        )

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(
            z_mu.get_device())  # get_device dziala tylko dla tensoróœ z gpu
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def reparameterize_for_cpu(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(
            'cpu')  # get_device dziala tylko dla tensoróœ z gpu
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize_for_cpu(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded


class VAE_Linear_0103(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 5),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(5),
            torch.nn.Linear(5, 4),
            #nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.z_mean = nn.Sequential(
            nn.Linear(4, 3)
        )

        self.z_log_var = nn.Sequential(
            nn.Linear(4, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            #nn.BatchNorm1d(4),
            nn.Linear(4, 5),
            nn.ReLU(),
            #nn.BatchNorm1d(5),
            nn.Linear(5, 6)
        )

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(
            z_mu.get_device())  # get_device dziala tylko dla tensoróœ z gpu
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def reparameterize_for_cpu(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(
            'cpu')  # get_device dziala tylko dla tensoróœ z gpu
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

class VAE_Linear_0203(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(5),
            torch.nn.Linear(5, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.z_mean = nn.Sequential(
            nn.Linear(4, 3)
        )

        self.z_log_var = nn.Sequential(
            nn.Linear(4, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.BatchNorm1d(5),
            nn.Linear(5, 6)
        )

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(
            z_mu.get_device())  # get_device dziala tylko dla tensoróœ z gpu
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def reparameterize_for_cpu(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(
            'cpu')  # get_device dziala tylko dla tensoróœ z gpu
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded