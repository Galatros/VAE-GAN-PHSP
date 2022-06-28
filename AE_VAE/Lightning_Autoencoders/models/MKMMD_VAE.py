import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Lit_MKMMD_VAE(pl.LightningModule):
    def __init__(self, learning_rate: float = 0.0005, name: str = 'MKMMD_VAE', kernel_mul: float = 2.0, kernel_num: int = 5, fix_sigma: float = None, weight_decay: float = 0.0, scheduler_gamma: float = 0.95):
        super(Lit_MKMMD_VAE, self).__init__()

        self.name = name
        self.learning_rate = learning_rate
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Linear(6, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            torch.nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
        )

        self.z_mean = nn.Sequential(
            nn.Linear(400, 12),
            nn.LeakyReLU(),
            nn.BatchNorm1d(12),
            nn.Linear(12, 4)

        )

        self.z_log_var = nn.Sequential(
            nn.Linear(400, 12),
            nn.LeakyReLU(),
            nn.BatchNorm1d(12),
            nn.Linear(12, 4)
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(),
            nn.Linear(12, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 6)
        )

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1), device=self.device)
        z = z_mu + eps * torch.exp(z_log_var/2.)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul**i)
                              for i in range(kernel_num)]
            kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                          for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def mkmmd_loss_function(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

    def training_step(self, batch, batch_idx):
        features = batch
        # Forward pass
        encoded, _, _, decoded = self(features)

        batchsize = batch.size(0)

        mkmmd_loss = self.mkmmd_loss_function(
            source=encoded, target=torch.randn_like(encoded, device=self.device))
        mse_loss = F.mse_loss(features, decoded, reduction='none')
        mse_loss = mse_loss.view(batchsize, -1).sum(axis=1)
        mse_loss = mse_loss.mean()

        combined_loss = mse_loss+mkmmd_loss

        train_logs = {"combined_loss": combined_loss.detach(
        ), "mse_loss": mse_loss.detach(), "mkmmd_loss": mkmmd_loss.detach()}

        self.log("train_logs", train_logs, on_step=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        features = batch

        # Forward pass
        encoded, _, _, decoded = self(features)
        batchsize = batch.size(0)

        mkmmd_loss = self.mkmmd_loss_function(
            source=encoded, target=torch.randn_like(encoded, device=self.device))
        mse_loss = F.mse_loss(features, decoded, reduction='none')
        mse_loss = mse_loss.view(batchsize, -1).sum(axis=1)
        mse_loss = mse_loss.mean()

        combined_loss = mse_loss+mkmmd_loss

        validation_logs = {"combined_loss": combined_loss.detach(
        ), "mse_loss": mse_loss.detach(), "mkmmd_loss": mkmmd_loss.detach()}
        self.log("validation_logs", validation_logs)
        return combined_loss.detach()

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x for x in outputs]).mean()
        val_epoch_end_logs = {"avg_val_loss": avg_loss.detach()}
        # use key 'log'
        self.log("validation_epoch_end_logs", val_epoch_end_logs)
        return avg_loss.detach()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]
