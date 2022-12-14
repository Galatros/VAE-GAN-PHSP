import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import Tensor


class Lit_MKMMD_InfoVAE(pl.LightningModule):
    def __init__(self, learning_rate=0.0005, name='MKMMD_InfoVAE', mkmmd_weight=110,  kernel_mul=2.0, kernel_num=5, fix_sigma=None, kld_weight=-9.0, reconstruction_weight=10.5, bias_correction_term=0.00131, weight_decay=0.0, scheduler_gamma=0.95):
        super(Lit_MKMMD_InfoVAE, self).__init__()

        self.name = name
        self.learning_rate = learning_rate
        self.mkmmd_weight = mkmmd_weight
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.kld_weight = kld_weight
        self.reconstruction_weight = reconstruction_weight
        self.bias_correction_term = bias_correction_term  # batch_size/num_of_photons
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

    def mkmmd_info_loss_function(self, decoded, input, encoded, z_mean, z_log_var) -> dict:

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)

        recons_loss = F.mse_loss(decoded, input)
        mkmmd_loss = self.mkmmd_loss_function(
            source=encoded, target=torch.randn_like(encoded, device=self.device))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var -
                              z_mean ** 2 - z_log_var.exp(), dim=1), dim=0)

        combined_loss = self.reconstruction_weight * recons_loss + \
            (1. - self.kld_weight) * self.bias_correction_term * kld_loss + \
            (self.kld_weight + self.mkmmd_weight - 1.)/bias_corr * mkmmd_loss

        return combined_loss, recons_loss, mkmmd_loss, -kld_loss

    # def compute_kernel(self,
    #                    x1: Tensor,
    #                    x2: Tensor) -> Tensor:
    #     # Convert the tensors into row and column vectors
    #     D = x1.size(1)
    #     N = x1.size(0)

    #     x1 = x1.unsqueeze(-2) # Make it into a column tensor
    #     x2 = x2.unsqueeze(-3) # Make it into a row tensor

    #     """
    #     Usually the below lines are not required, especially in our case,
    #     but this is useful when x1 and x2 have different sizes
    #     along the 0th dimension.
    #     """
    #     x1 = x1.expand(N, N, D)
    #     x2 = x2.expand(N, N, D)

    #     if self.kernel_type == 'rbf':
    #         result = self.compute_rbf(x1, x2)
    #     elif self.kernel_type == 'imq':
    #         result = self.compute_inv_mult_quad(x1, x2)
    #     else:
    #         raise ValueError('Undefined kernel type.')

    #     return result

    # def compute_rbf(self,
    #                 x1: Tensor,
    #                 x2: Tensor,
    #                 eps: float = 1e-7) -> Tensor:
    #     """
    #     Computes the RBF Kernel between x1 and x2.
    #     :param x1: (Tensor)
    #     :param x2: (Tensor)
    #     :param eps: (Float)
    #     :return:
    #     """
    #     z_dim = x2.size(-1)
    #     sigma = 2. * z_dim * self.lantent_var

    #     result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
    #     return result

    # def compute_inv_mult_quad(self,
    #                            x1: Tensor,
    #                            x2: Tensor,
    #                            eps: float = 1e-7) -> Tensor:
    #     """
    #     Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
    #     given by
    #             k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
    #     :param x1: (Tensor)
    #     :param x2: (Tensor)
    #     :param eps: (Float)
    #     :return:
    #     """
    #     z_dim = x2.size(-1)
    #     C = 2 * z_dim * self.lantent_var
    #     kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

    #     # Exclude diagonal elements
    #     result = kernel.sum() - kernel.diag().sum()

    #     return result

    # def compute_mmd(self, z: Tensor) -> Tensor:
    #     # Sample from prior (Gaussian) distribution
    #     prior_z = torch.randn_like(z, device=self.device)

    #     prior_z__kernel = self.compute_kernel(prior_z, prior_z)
    #     z__kernel = self.compute_kernel(z, z)
    #     priorz_z__kernel = self.compute_kernel(prior_z, z)

    #     mmd = prior_z__kernel.mean() + \
    #           z__kernel.mean() - \
    #           2 * priorz_z__kernel.mean()
    #     return mmd

    def training_step(self, batch, batch_idx):
        features = batch
        # Forward pass
        encoded, z_mean, z_log_var, decoded = self(features)
        # kldiv_loss = -0.5 * torch.sum(1 + z_log_var -
        #                               z_mean**2 - torch.exp(z_log_var), axis=1)
        # batchsize = batch.size(0)
        # kldiv_loss = kldiv_loss.mean()
        combined_loss, mse_loss, mkmmd_loss, kld_loss = self.mkmmd_info_loss_function(
            decoded=decoded, encoded=encoded, z_mean=z_mean, z_log_var=z_log_var, input=batch)
        # mse_loss = F.mse_loss(features, decoded, reduction='none')
        # mse_loss = mse_loss.view(batchsize,-1).sum(axis=1)
        # mse_loss=mse_loss.mean()

        # combined_loss=mse_loss+mkmmd_loss

        train_logs = {"combined_loss": combined_loss.detach(), "mse_loss": mse_loss.detach(
        ), "mkmmd_loss": mkmmd_loss.detach(), "kld_loss": kld_loss.detach()}
        # use key 'log'
        # return {"loss": combined_loss, 'log': tensorboard_logs}
        self.log("train_logs", train_logs, on_step=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        features = batch

        # Forward pass
        encoded, z_mean, z_log_var, decoded = self(features)

        # kldiv_loss = -0.5 * torch.sum(1 + z_log_var -
        #                               z_mean**2 - torch.exp(z_log_var), axis=1)
        # batchsize = kldiv_loss.size(0)
        # kldiv_loss = kldiv_loss.mean()
        # batchsize = batch.size(0)
        # kldiv_loss = kldiv_loss.mean()
        combined_loss, mse_loss, mkmmd_loss, kld_loss = self.mkmmd_info_loss_function(
            decoded=decoded, encoded=encoded, z_mean=z_mean, z_log_var=z_log_var, input=batch)
        # mse_loss = F.mse_loss(features, decoded, reduction='none')
        # mse_loss = mse_loss.view(batchsize,-1).sum(axis=1)
        # mse_loss=mse_loss.mean()

        # combined_loss=mse_loss+mkmmd_loss

        validation_logs = {"combined_loss": combined_loss.detach(), "mse_loss": mse_loss.detach(
        ), "mkmmd_loss": mkmmd_loss.detach(), "kld_loss": kld_loss.detach()}
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
