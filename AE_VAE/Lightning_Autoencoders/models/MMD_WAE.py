import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor


class Lit_MMD_WAE(pl.LightningModule):
    # kernel_mul=2.0, kernel_num=5, fix_sigma=None
    def __init__(self, learning_rate=0.0005, name='MMD_WAE', mmd_weight=100,  kernel_type='imq', lantent_var=2.0, scheduler_gamma=0.95, weight_decay=0.0):
        super(Lit_MMD_WAE, self).__init__()

        self.name = name
        self.learning_rate = learning_rate
        self.mmd_weight = mmd_weight
        self.kernel_type = kernel_type
        self.lantent_var = lantent_var
        self.scheduler_gamma = scheduler_gamma
        self.weight_decay = weight_decay
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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def mmdwae_loss_function(self, decoded, input, encoded) -> dict:

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        mmd_weight = self.mmd_weight/bias_corr

        recons_loss = F.mse_loss(decoded, input)
        mmd_loss = self.compute_mmd(encoded, mmd_weight)

        combined_loss = recons_loss + mmd_loss

        return combined_loss, recons_loss, mmd_loss

    def compute_kernel(self,
                       x1: Tensor,
                       x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result

    def compute_rbf(self,
                    x1: Tensor,
                    x2: Tensor,
                    eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.lantent_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,
                              x1: Tensor,
                              x2: Tensor,
                              eps: float = 1e-7) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by
                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.lantent_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: Tensor, mmd_weight: float) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z, device=self.device)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = mmd_weight * prior_z__kernel.mean() + \
            mmd_weight * z__kernel.mean() - \
            mmd_weight * 2 * priorz_z__kernel.mean()
        return mmd

    def training_step(self, batch, batch_idx):
        features = batch
        # Forward pass
        encoded, decoded = self(features)

        combined_loss, mse_loss, mmd_loss = self.mmdwae_loss_function(
            decoded=decoded, encoded=encoded, input=batch)

        train_logs = {"combined_loss": combined_loss.detach(
        ), "mse_loss": mse_loss.detach(), "mmd_loss": mmd_loss.detach()}
        # use key 'log'
        # return {"loss": combined_loss, 'log': tensorboard_logs}
        self.log("train_logs", train_logs, on_step=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        features = batch

        # Forward pass
        encoded, decoded = self(features)

        combined_loss, mse_loss, mmd_loss = self.mmdwae_loss_function(
            decoded=decoded, encoded=encoded, input=batch)

        validation_logs = {"combined_loss": combined_loss.detach(
        ), "mse_loss": mse_loss.detach(), "mmd_loss": mmd_loss.detach()}
        self.log("validation_logs", validation_logs)
        return combined_loss.detach()
        # return {"val_loss": combined_loss.detach(), 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_loss = torch.stack([x for x in outputs]).mean()
        val_epoch_end_logs = {"avg_val_loss": avg_loss.detach()}
        # use key 'log'
        self.log("validation_epoch_end_logs", val_epoch_end_logs)
        return avg_loss.detach()
        # return {'val_loss': avg_loss.detach(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]
