import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import Tensor
from torch import distributions as dist


class Lit_S_WAE(pl.LightningModule):
    def __init__(self, learning_rate=0.0005, name='S_WAE', latent_dim=4, w_weight=110, bias_correction_term=0.00131, wasserstein_deg=2.0,  num_projections=50, projection_dist='normal', weight_decay=0.0, scheduler_gamma=0.95):
        super(Lit_S_WAE, self).__init__()

        self.name = name
        self.learning_rate = learning_rate
        self.w_weight = w_weight
        self.wasserstein_deg = wasserstein_deg
        self.num_projections = num_projections
        self.latent_dim = latent_dim
        self.projection_dist = projection_dist
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
            nn.Linear(400, 12),
            nn.LeakyReLU(),
            nn.BatchNorm1d(12),
            nn.Linear(12, self.latent_dim)
        )

        # self.z_mean = nn.Sequential(
        #     nn.Linear(400, 12),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(12),
        #     nn.Linear(12,self.latent_dim)

        # )

        # self.z_log_var = nn.Sequential(
        #     nn.Linear(400, 12),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(12),
        #     nn.Linear(12, self.latent_dim)
        # )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 12),
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

    # def reparameterize(self, z_mu, z_log_var):
    #     eps = torch.randn(z_mu.size(0), z_mu.size(1), device=self.device)
    #     z = z_mu + eps * torch.exp(z_log_var/2.)
    #     return z

    def forward(self, x):
        encoded = self.encoder(x)
        # z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        # encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, None, None, decoded

    def s_wae_loss_function(self, decoded, input, encoded) -> dict:

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.w_weight / bias_corr

        recons_loss_l2 = F.mse_loss(decoded, input)
        recons_loss_l1 = F.l1_loss(decoded, input)
        recons_loss = recons_loss_l1+recons_loss_l2

        s_wd_loss = self.compute_s_wd(
            encoded=encoded, w_weight=self.w_weight, reg_weight=reg_weight)

        combined_loss = recons_loss + s_wd_loss

        return combined_loss, recons_loss, s_wd_loss

    def get_random_projections(self, latent_dim: int, num_samples: int) -> Tensor:
        """
        Returns random samples from latent distribution's (Gaussian)
        unit sphere for projecting the encoded samples and the
        distribution samples.
        :param latent_dim: (Int) Dimensionality of the latent space (D)
        :param num_samples: (Int) Number of samples required (S)
        :return: Random projections from the latent unit sphere
        """
        if self.projection_dist == 'normal':
            rand_samples = torch.randn(
                num_samples, latent_dim, device=self.device)
        elif self.projection_dist == 'cauchy':
            rand_samples = dist.Cauchy(torch.tensor([0.0]),
                                       torch.tensor([1.0])).sample((num_samples, latent_dim)).squeeze()
        else:
            raise ValueError('Unknown projection distribution.')

        rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
        return rand_proj  # [S x D]

    def compute_s_wd(self,
                     encoded: Tensor,
                     w_weight: float,
                     reg_weight: float) -> Tensor:
        """
        Computes the Sliced Wasserstein Distance (SWD) - which consists of
        randomly projecting the encoded and prior vectors and computing
        their Wasserstein distance along those projections.
        :param z: Latent samples # [N  x D]
        :param p: Value for the p^th Wasserstein distance
        :param reg_weight:
        :return:
        """
        prior_z = torch.randn_like(encoded, device=self.device)  # [N x D]

        proj_matrix = self.get_random_projections(self.latent_dim,
                                                  num_samples=self.num_projections).transpose(0, 1)

        latent_projections = encoded.matmul(proj_matrix)  # [N x S]
        prior_projections = prior_z.matmul(proj_matrix)  # [N x S]

        # The Wasserstein distance is computed by sorting the two projections
        # across the batches and computing their element-wise l2 distance
        w_dist = torch.sort(latent_projections.t(), dim=1)[0] - \
            torch.sort(prior_projections.t(), dim=1)[0]
        w_dist = w_dist.pow(w_weight)
        return reg_weight * w_dist.mean()

    def training_step(self, batch, batch_idx):
        # Forward pass
        encoded, _, _, decoded = self(batch)

        combined_loss, recons_loss, s_wd_loss = self.s_wae_loss_function(
            decoded=decoded, encoded=encoded, input=batch)

        train_logs = {"combined_loss": combined_loss.detach(
        ), "recons_loss": recons_loss.detach(), "s_wd_loss": s_wd_loss.detach()}
        self.log("train_logs", train_logs, on_step=True)
        return combined_loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        encoded, _, _, decoded = self(batch)

        combined_loss, recons_loss, s_wd_loss = self.s_wae_loss_function(
            decoded=decoded, encoded=encoded, input=batch)

        validation_logs = {"combined_loss": combined_loss.detach(
        ), "recons_loss": recons_loss.detach(), "s_wd_loss": s_wd_loss.detach()}
        self.log("validation_logs", validation_logs)
        return combined_loss.detach()

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x for x in outputs]).mean()
        val_epoch_end_logs = {"avg_val_loss": avg_loss.detach()}
        self.log("validation_epoch_end_logs", val_epoch_end_logs)
        return avg_loss.detach()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]
