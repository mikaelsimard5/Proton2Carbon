import ast
import os
import kornia as K
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import lightning as L
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from monai.networks.nets import UNet, BasicUNet, BasicUNetPlusPlus

from Utils import Loss_Functions
from Utils.Initialisation import init_weights


class Unet(L.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        if config["BASEMODEL"]["Loss_Function"] == "WeightedMSELoss":
            self.loss_fcn = Loss_Functions.WeightedMSELoss()
        else:
            self.loss_fcn = getattr(torch.nn, config["BASEMODEL"]["Loss_Function"])()

        # See https://docs.monai.io/en/stable/networks.html#basicunet to tweak the network.
        if config['ADVANCEDMODEL']['Name'] == "BasicUNet":
            self.model = BasicUNet(spatial_dims=2,
                                in_channels=1,
                                out_channels=1,
                                act=config['ADVANCEDMODEL']['Activation'],
                                dropout=config['ADVANCEDMODEL']['Drop_Rate'])
        elif config['ADVANCEDMODEL']['Name'] == "BasicUnetPlusPlus":
            self.model = BasicUNetPlusPlus(spatial_dims=2,
                                in_channels=1,
                                out_channels=1,
                                act=config['ADVANCEDMODEL']['Activation'],
                                dropout=config['ADVANCEDMODEL']['Drop_Rate'],
                                features=self.config['ADVANCEDMODEL']['Unet_features'])
            init_weights(self.model, init_type = "xavier")
        else:
            raise ValueError(f"Unknown model:{config['ADVANCEDMODEL']['Name']}.")



    def forward(self, x):
        # Forward pass through U-Net

        if self.config['ADVANCEDMODEL']['Name'] == "BasicUNet":
            return self.model(x)
        elif self.conI tfig['ADVANCEDMODEL']['Name'] == "BasicUnetPlusPlus":
            x = self.model(x)
            if len(x)>1:
                raise ValueError('Not sure why this is len > 1.')
            return x[0] # not sure why it's in a list...

    def on_after_batch_transfer(self, batch, dataloader_idx):
        proton_image, carbon_image = batch

        kernel_size = self.config['AUGMENTATIONS']['Median_Kernel_Size']
        proton_image = K.filters.median_blur(proton_image, kernel_size)
        proton_image = K.enhance.normalize_min_max(proton_image, min_val=0.0, max_val=1.0)

        if not self.config['ADVANCEDMODEL']['Inference']:
            carbon_image = K.filters.median_blur(carbon_image, kernel_size)
            carbon_image = K.enhance.normalize_min_max(carbon_image, min_val=0.0, max_val=1.0)

        # Augmentations to perform only on the training data
        if self.training:
            for k_transform, v_args in self.config['AUGMENTATIONS_KORNIA'].items():
                transform_class = getattr(K.augmentation, k_transform)
                cur_transform = transform_class(**ast.literal_eval(v_args))
                proton_image = cur_transform(proton_image)


        proton_image = K.enhance.normalize_min_max(proton_image, min_val=0.0, max_val=1.0)

        return proton_image, carbon_image


    def training_step(self, train_batch, batch_idx):
        proton_image, carbon_image = train_batch
        predicted_carbon_image = self.forward(proton_image)

        loss = self.loss_fcn(predicted_carbon_image, carbon_image)

        self.log(f'train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        proton_image, carbon_image = val_batch
        predicted_carbon_image = self.forward(proton_image)
        loss = self.loss_fcn(predicted_carbon_image, carbon_image)

        # every 100 steps, save 
        if batch_idx % 100 == 0:
            self.plot_validation_example(proton_image, carbon_image, predicted_carbon_image, batch_idx)

        self.log(f'val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss        

    def test_step(self, test_batch, batch_idx):
        proton_image, carbon_image = test_batch
        predicted_carbon_image = self.forward(proton_image)
        loss = self.loss_fcn(predicted_carbon_image, carbon_image)

        self.log(f'test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss   

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        image = batch
        predicted_carbon_image = self(image)  # Unpack logit and attention maps
        return predicted_carbon_image

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config['OPTIMIZER']['Algorithm'])
        optimizer = optimizer(self.parameters(),
                              lr=self.config["OPTIMIZER"]["lr"],
                              eps=self.config["OPTIMIZER"]["eps"],
                              betas=(0.9, 0.999),
                              weight_decay=self.config['REGULARIZATION']['Optimizer_Weight_Decay'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['ADVANCEDMODEL']['Max_Epochs'])

        return [optimizer], [scheduler]

    def plot_validation_example(self, proton_image, carbon_image, predicted_carbon_image, batch_idx):

        # Prepare data
        proton_img = proton_image[0, 0, :, :].cpu().numpy()
        carbon_img = carbon_image[0, 0, :, :].cpu().numpy()
        output_img = predicted_carbon_image[0, 0, :, :].cpu().numpy()
        abs_diff = torch.abs(carbon_image - predicted_carbon_image)[0, 0, :, :].cpu().numpy()
        rel_error = torch.abs(carbon_image - predicted_carbon_image) / torch.abs(carbon_image)
        rel_error = rel_error[0, 0, :, :].cpu().numpy()

        vmin, vmax = carbon_img.min(), carbon_img.max()

        # Set up the subplot
        fig, axs = plt.subplots(2, 2, figsize=(15, 6))

        # First plot
        cax1 = axs[0, 0].imshow(carbon_img, cmap='gray', vmin=vmin, vmax=vmax)
        axs[0, 0].set_title('Label (carbon image)')
        fig.colorbar(cax1, ax=axs[0, 0])

        # Second plot
        cax2 = axs[0, 1].imshow(output_img, cmap='gray', vmin=vmin, vmax=vmax)
        axs[0, 1].set_title('Prediction')
        fig.colorbar(cax2, ax=axs[0, 1])

        # Third plot (Absolute Difference)
        cax3 = axs[1, 0].imshow(abs_diff, cmap='gray')
        axs[1, 0].set_title('Absolute Difference')
        fig.colorbar(cax3, ax=axs[1, 0])

        # Fourth plot (proton image)
        cax4 = axs[1, 1].imshow(proton_img, cmap='gray', vmin=proton_img.min(), vmax=proton_img.max())
        axs[1, 1].set_title('Original (proton image)')
        fig.colorbar(cax4, ax=axs[1, 1])

        # Adjust layout
        plt.tight_layout()

        # Save the image
        epoch_num = self.current_epoch
        save_path = os.path.join(self.config['CHECKPOINT']['logger_folder'], "example_images", "predictions", f"epoch_{epoch_num}")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"image_0_{batch_idx}.png"))

        plt.close(fig)  # Close the figure to free memory

