import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generate = nn.Sequential(
            nn.Linear(self.latent_dim, 400),
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
            torch.nn.Linear(400, 6)
        )

    def forward(self, x):
        return self.generate(x)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.discriminate = nn.Sequential(
            nn.Linear(6, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminate(x)


class Lit_WGAN(pl.LightningModule):
    def __init__(self, name: str = 'WGAN', latent_dim: int = 6, learning_rate: float = 0.0005, critic_iterations: int = 5, weight_clip: float = 0.01, weight_decay: float = 0.0, scheduler_gamma: float = 0.95):
        super(Lit_WGAN, self).__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.critic_iterations = critic_iterations
        self.weight_clip = weight_clip
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.latent_dim)
        self.critic = Critic()

    def forward(self, x):
        return self.generator(x)

    # def adversarial_loss(self, y_hat, y):
    #     return F.binary_cross_entropy(y_hat,y)

    def generator_step(self, noise):
        g_loss = -torch.mean(self.critic(self(noise)))

        train_logs = {"g_loss": g_loss.detach()}

        self.log("train_g_logs", train_logs, on_step=True, prog_bar=True)
        return g_loss

        # # generate photons
        # self.generated_photons = self(noise)

        # # ground truth result (ie: all fake)
        # # put on GPU because we created this tensor inside training_loop
        # valid = torch.ones(photons_batch.size(0), 1)
        # valid = valid.type_as(photons_batch)

        # # adversarial loss is binary cross-entropy
        # g_loss = self.adversarial_loss(self.critic(self(noise)), valid)
        # train_logs = {"g_loss": g_loss.detach()}

        # self.log("train_g_logs", train_logs, on_step=True)
        # return g_loss

    def critic_step(self, noise, photons_batch):
        fake_photons = self(noise)
        critic_real = self.critic(photons_batch)
        critic_fake = self.critic(fake_photons)
        critic_loss = -(torch.mean(critic_real) - torch.mean(critic_fake))
        # self.critic.zero_grad()
        # loss_critic.backward(retain_graph=True)
        # opt_critic.step()

        # for p in self.critic.parameters():
        #         p.data.clamp_(-self.weight_clip, self.weight_clip)

        train_logs = {"c_loss": critic_loss.detach()}

        self.log("train_c_logs", train_logs, on_step=True, prog_bar=True)
        return critic_loss

        # # Measure discriminator's ability to classify real from generated samples

        # # how well can it label as real?
        # valid = torch.ones(photons_batch.size(0), 1)
        # valid = valid.type_as(photons_batch)

        # real_loss = self.adversarial_loss(self.critic(photons_batch), valid)

        # # how well can it label as fake?
        # fake = torch.zeros(photons_batch.size(0), 1)
        # fake = fake.type_as(photons_batch)

        # fake_loss = self.adversarial_loss(self.critic(self(noise).detach()), fake)

        # # discriminator loss is the average of these
        # d_loss = (real_loss + fake_loss) / 2
        # train_logs = {"d_loss": d_loss.detach()}

        # self.log("train_d_logs", train_logs, on_step=True)
        # return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        photons_batch = batch

        # Forward pass
        # sample noise
        noise = torch.randn(photons_batch.shape[0], self.latent_dim)
        noise = noise.type_as(photons_batch)

        # train discriminator
        if optimizer_idx == 1:
            return self.critic_step(noise=noise, photons_batch=photons_batch)

        # train generator
        if optimizer_idx == 0:
            return self.generator_step(noise=noise)

    # def validation_step(self, batch, _):
    #     features = batch

    #     # Forward pass
    #     _, z_mean, z_log_var, decoded = self(features)

    #     kld_loss = -0.5 * torch.sum(1 + z_log_var -
    #                                   z_mean**2 - torch.exp(z_log_var), axis=1)
    #     batchsize = kld_loss.size(0)
    #     kld_loss = kld_loss.mean()

    #     mse_loss = F.mse_loss(features, decoded, reduction='none')
    #     mse_loss = mse_loss.view(batchsize,-1).sum(axis=1)
    #     mse_loss=mse_loss.mean()

    #     combined_loss=mse_loss+self.beta_weight*kld_loss

    #     validation_logs = {"combined_loss": combined_loss.detach(), "mse_loss": mse_loss.detach(), "kld_loss":kld_loss.detach()}
    #     self.log("validation_logs", validation_logs)
    #     return combined_loss.detach()

    # def validation_epoch_end(self, outputs):
    #     # outputs = list of dictionaries
    #     # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_loss = torch.stack([x for x in outputs]).mean()
    #     val_epoch_end_logs = {"avg_val_loss": avg_loss.detach()}
    #     self.log("validation_epoch_end_logs", val_epoch_end_logs)
    #     return avg_loss.detach()

    def configure_optimizers(self):
        optimizer_g = torch.optim.RMSprop(self.generator.parameters(
        ), lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer_c = torch.optim.RMSprop(self.critic.parameters(
        ), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_g, gamma=self.scheduler_gamma)
        scheduler_c = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_c, gamma=self.scheduler_gamma)
        return [optimizer_g, optimizer_c], [scheduler_g, scheduler_c]

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False):

        # update critic every step
        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)
            for p in self.critic.parameters():
                p.data.clamp_(-self.weight_clip, self.weight_clip)

        # update generator every 4 steps
        if optimizer_idx == 0:
            if (batch_idx + 1) % self.critic_iterations == 0:
                # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                optimizer.step(closure=optimizer_closure)
            else:
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()
