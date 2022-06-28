import os
from abc import ABC

import pytorch_lightning as pl
import torchvision.utils as vutils
from torch import optim
import numpy as np
# from . import BaseVAE
from . import *


class VAELightningModule(pl.LightningModule, ABC):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAELightningModule, self).__init__()
        self.save_hyperparameters()  # Added by me, to test

        self.model = vae_model
        self.name = 'VAE'
        self.params = params
        self.curr_device = torch.device('cpu')
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
    def get_latent_shape(self):
        return tuple((1, 128))

    def get_latent_dims(self):
        return np.prod(self.get_latent_shape())

    def set_output_class(self, new_class):
        self.outclass = new_class

    def partial_forward(self, input, layer_name):
        # doing forward at the expense of performance
        return self.forward(input)

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        z = torch.randn(n_samples, self.model.latent_dim).to(self.curr_device)
        return z

    def get_max_latents(self):
        return 1

    def sample_np(self, z=None, n_samples=1, seed=None):
        if z is None:
            z = self.sample_latent(n_samples, seed=seed)
        elif isinstance(z, list):
            # print(z)
            z = z[0] # To fix error in visualize.py line 88. out_batch = create_strip_centered(inst, edit_type, layer_key, [latent],
            # z = [torch.tensor(l).to(self.device) if not torch.is_tensor(l) else l for l in z]
        elif not torch.is_tensor(z):
            z = torch.tensor(z).to(self.device)
        img = self.forward(z)
        img_np = img.permute(0, 2, 3, 1).cpu().detach().numpy()
        return np.clip(img_np, 0.0, 1.0).squeeze()
    def latent_space_name(self):
        return 'Z'
    # Just using decoder
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model.decode(input)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
