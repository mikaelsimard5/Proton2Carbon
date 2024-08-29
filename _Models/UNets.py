import ast
import json
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
        self.kernel_size = self.config['AUGMENTATIONS']['median_kernel_size']
        self.laplacian_kernel_size = 3 # 3x3 around the pixel for derivative

        if config["BASEMODEL"]["loss_function"] == "WeightedMSELoss":
            self.loss_fcn = Loss_Functions.WeightedMSELoss()
            print('Using custom loss function...')
        elif config["BASEMODEL"]["loss_function"] == "GradientWeightedMSELoss":
            self.loss_fcn = Loss_Functions.GradientWeightedMSELoss()
            print('Using custom loss function...')            
        else:
            self.loss_fcn = getattr(torch.nn, config["BASEMODEL"]["loss_function"])()

        # See https://docs.monai.io/en/stable/networks.html#basicunet to tweak the network.
        if config['ADVANCEDMODEL']['name'] == "BasicUNet":
            self.model = BasicUNet(spatial_dims=2,
                                in_channels=1,
                                out_channels=1,
                                act=config['ADVANCEDMODEL']['activation'],
                                dropout=config['ADVANCEDMODEL']['drop_rate'])
        elif config['ADVANCEDMODEL']['name'] == "BasicUnetPlusPlus":
            self.model = BasicUNetPlusPlus(spatial_dims=2,
                                in_channels=1,
                                out_channels=1,
                                act=config['ADVANCEDMODEL']['activation'],
                                dropout=config['ADVANCEDMODEL']['drop_rate'],
                                features=self.config['ADVANCEDMODEL']['unet_features'])
            init_weights(self.model, init_type = "xavier")
        else:
            raise ValueError(f"Unknown model:{config['ADVANCEDMODEL']['name']}.")


    def forward(self, x):
        # Forward pass through U-Net

        if self.config['ADVANCEDMODEL']['name'] == "BasicUNet":
            return self.model(x)
        elif self.config['ADVANCEDMODEL']['name'] == "BasicUnetPlusPlus":

            return self.model(x)[0] # for some reason it's in a list and we need the first element
        

    def curate_BP(self, batch_images):
        # Input should be a batch of images of shape (B, 1, H, W).

        # Blur the BP
        batch_images = K.filters.median_blur(batch_images, self.kernel_size)

        Profile = torch.sum(batch_images, dim=-1)  # Shape: (B, 1, H)

        # Compute the background level
        median_profile = torch.median(Profile, dim=-1).values  # Shape: (B, 1)
        percentile_25_profile = torch.quantile(Profile, 0.25, dim=-1)  # Shape: (B, 1)
        background_level = torch.mean(torch.stack([median_profile, percentile_25_profile], dim=-1), dim=-1)  # Shape: (B, 1)
        
        # Calculate prominence
        max_profile = torch.max(Profile, dim=-1).values  # Shape: (B, 1)
        prominence = max_profile - background_level  # Shape: (B, 1)
        
        # Define the relevant data based on the threshold
        valid_lat_index = Profile > (background_level.unsqueeze(-1) + 0.05 * prominence.unsqueeze(-1))  # Shape: (B, 1, H)
        
        # Sum along the valid latitudinal indices
        BP = torch.sum(batch_images * valid_lat_index.unsqueeze(-1), dim=2)  # Shape: (B, 1, W)
        
        # Normalize the BP - not useful here
        #BP_min = BP.min(dim=-1, keepdim=True).values  # Shape: (B, 1, 1)
        #BP_max = BP.max(dim=-1, keepdim=True).values  # Shape: (B, 1, 1)
        #BP = (BP - BP_min) / (BP_max - BP_min)  # Shape: (B, 1, W)   

        return BP     


    def on_after_batch_transfer(self, batch, dataloader_idx):
        # Define some transforms to be done on GPU for efficiency.

        proton_image, carbon_image = batch

        kernel_size = self.kernel_size
        #proton_image = K.filters.median_blur(proton_image, kernel_size)
        proton_image = K.enhance.normalize_min_max(proton_image, min_val=0.0, max_val=1.0)

        if not self.config['ADVANCEDMODEL']['inference']:
            #carbon_image = K.filters.median_blur(carbon_image, kernel_size)
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
        #loss = self.loss_fcn(predicted_carbon_image, carbon_image)

        # Loss calculated on profiles instead
        
        predicted_carbon_profiles = self.curate_BP(predicted_carbon_image)
        true_carbon_profiles = self.curate_BP(carbon_image)
        loss = self.loss_fcn(predicted_carbon_profiles, true_carbon_profiles)
        #print(predicted_carbon_profiles, true_carbon_profiles)
        #print(loss)

        # Pixel + Gradient loss
        #predicted_carbon_grad = K.filters.laplacian(predicted_carbon_image, self.laplacian_kernel_size, normalized=True)
        #true_carbon_grad = K.filters.laplacian(carbon_image, self.laplacian_kernel_size, normalized=True)
        #loss = self.loss_fcn(predicted_carbon_image, carbon_image) + self.loss_fcn(predicted_carbon_grad, true_carbon_grad)

        self.log(f'train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        proton_image, carbon_image = val_batch
        predicted_carbon_image = self.forward(proton_image)
        #loss = self.loss_fcn(predicted_carbon_image, carbon_image)

        # Loss calculated on profiles instead
        predicted_carbon_profiles = self.curate_BP(predicted_carbon_image)
        true_carbon_profiles = self.curate_BP(carbon_image)
        loss = self.loss_fcn(predicted_carbon_profiles, true_carbon_profiles)        

        # Pixel + Gradient loss
        #predicted_carbon_grad = K.filters.laplacian(predicted_carbon_image, self.laplacian_kernel_size, normalized=True)
        #true_carbon_grad = K.filters.laplacian(carbon_image, self.laplacian_kernel_size, normalized=True)
        #loss = self.loss_fcn(predicted_carbon_image, carbon_image) + self.loss_fcn(predicted_carbon_grad, true_carbon_grad)        

        # every 100 steps, save 
        if batch_idx % 100 == 0:
            self.plot_validation_example(proton_image, carbon_image, predicted_carbon_image, batch_idx)

        self.log(f'val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss        

    def test_step(self, test_batch, batch_idx):
        proton_image, carbon_image = test_batch
        predicted_carbon_image = self.forward(proton_image)
        #loss = self.loss_fcn(predicted_carbon_image, carbon_image)

        # Loss calculated on profiles instead
        predicted_carbon_profiles = self.curate_BP(predicted_carbon_image)
        true_carbon_profiles = self.curate_BP(carbon_image)

        loss = self.loss_fcn(predicted_carbon_profiles, true_carbon_profiles)        

        # Pixel + Gradient loss
        #predicted_carbon_grad = K.filters.laplacian(predicted_carbon_image, self.laplacian_kernel_size, normalized=True)
        #true_carbon_grad = K.filters.laplacian(carbon_image, self.laplacian_kernel_size, normalized=True)
        #loss = self.loss_fcn(predicted_carbon_image, carbon_image) + self.loss_fcn(predicted_carbon_grad, true_carbon_grad)        

        self.log(f'test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss   

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        # Predict
        proton_image, proton_image_path = batch
        predicted_carbon_image = self(proton_image)  # Unpack logit and attention maps

        # Export 
        for img, pth in zip(predicted_carbon_image, proton_image_path):

            # Adjust path for export
            for k, v in self.config['EXPORT']['name_change'].items():
                pth = pth.replace(k, v)
            os.makedirs(os.path.dirname(pth), exist_ok=True)
            pil_img = v2.functional.to_pil_image(img)  # Convert tensor to PIL image
            pil_img.save(pth)  # Save each image        
        return predicted_carbon_image, proton_image_path

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config['OPTIMIZER']['algorithm'])
        optimizer = optimizer(self.parameters(),
                              lr=self.config["OPTIMIZER"]["lr"],
                              eps=self.config["OPTIMIZER"]["eps"],
                              betas=(0.9, 0.999),
                              weight_decay=self.config['REGULARIZATION']['optimizer_weight_decay'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['ADVANCEDMODEL']['max_epochs'])

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


    def on_save_checkpoint(self, checkpoint):
        def convert_tensors(obj):
            if isinstance(obj, dict):
                return {key: convert_tensors(value) for key, value in obj.items()}
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            else:
                return obj

        serializable_config = convert_tensors(self.config)
        checkpoint['config'] = json.dumps(serializable_config, sort_keys=True, indent=4)

    @classmethod
    def read_config_from_checkpoint(cls, checkpoint_path):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        config_str = checkpoint.get('config', '{}')
        config = json.loads(config_str)
        return config
    

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):

        def convert_lists(obj):
            if isinstance(obj, dict):
                return {key: convert_lists(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return torch.tensor(obj)
            else:
                return obj

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # Extract and parse the config and LabelEncoder from the checkpoint
        config_str = checkpoint.get('config', '{}')  # Use an empty dict string as default.
        config = json.loads(config_str)

        # Create the model with the extracted config and LabelEncoder
        model = cls(config, *args, **kwargs)

        # Load the state dict
        model.load_state_dict(checkpoint['state_dict'])

        return model        