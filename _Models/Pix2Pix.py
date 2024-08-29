import ast
import json
import kornia as K
import lightning as L
import torch
import torch.nn as nn
from monai.networks.nets import UNet, BasicUNet, BasicUNetPlusPlus
import functools
import matplotlib.pyplot as plt
import os
import torch.nn.utils.parametrizations as param
from torchvision.transforms import v2

from Utils.Initialisation import init_weights

class Pix2PixLightning(L.LightningModule):
    """ Learns a transformation image A -> image B (proton -> carbon) """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generator = BasicUNetPlusPlus_Monai(config)
        self.discriminator = NLayerDiscriminator(input_nc=2, ndf=64, use_spectral_norm=config['ADVANCEDMODEL']['spectral_norm_D']) # input_nc is C_input + C_output = 2 for grayscale
        self.criterionL1 = torch.nn.L1Loss()
        self.lambda_L1 = config['REGULARIZATION']['l1_weight_g']
        self.criterionGAN = GANLoss(label_smoothing=config['REGULARIZATION']['label_smoothing'],
                                    gan_mode = self.config['ADVANCEDMODEL']['gan_mode']).to(self.device)  # defined here
        self.automatic_optimization = False # not supported in Lightning for multiple optimisers

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, x):
        return self.generator(x)

    def on_after_batch_transfer(self, batch, dataloader_idx): # perform transforms on GPU
        real_A, real_B = batch # A is proton, B is carbon
        k = self.config['AUGMENTATIONS']['median_kernel_size']

        # before: median blur done by default
        real_A = K.filters.median_blur(real_A, kernel_size=k)

        # Normalise [0,1] carbon data if we are not in inference mode.
        if not self.config['ADVANCEDMODEL']['inference']:
            real_B = K.filters.median_blur(real_B, kernel_size=k)
            real_B = K.enhance.normalize_min_max(real_B, min_val=0.0, max_val=1.0)        

        # Normalise [0,1] proton data
        real_A = K.enhance.normalize_min_max(real_A, min_val=0.0, max_val=1.0)
        #real_A_before_augment = real_A.clone().detach()
        
        # Data augment for proton images 
        if self.training:
            for k_transform, v_args in self.config['AUGMENTATIONS_KORNIA'].items():
                transform_class = getattr(K.augmentation, k_transform)
                cur_transform = transform_class(**ast.literal_eval(v_args))
                real_A = cur_transform(real_A)
        real_A = K.enhance.normalize_min_max(real_A, min_val=0.0, max_val=1.0)

        #if self.training:
            #self.plot_augmentation(real_A_before_augment, real_A)

        return real_A, real_B        

    def training_step(self, train_batch, batch_idx):
        # for manual optimisation, see https://lightning.ai/docs/pytorch/stable/common/optimization.html#optimization
        opt_G, opt_D = self.optimizers()
        sch_G, sch_D = self.lr_schedulers()

        # Compute fake image
        real_A, real_B = train_batch

        # -------------------- Update Discriminator --------------------
        # Enable backprop for the discriminator
        self.set_requires_grad(self.discriminator, True)

        opt_D.zero_grad() # zero gradients of discriminator

        # Forward pass for fake images and detach to prevent backprop to the generator
        fake_B = self.forward(real_A)
        fake_pair = torch.cat((real_A, fake_B), 1)
        pred_fake = self.discriminator(fake_pair.detach())
        loss_D_fake = self.criterionGAN(pred_fake, target_is_real=False)

        # Forward pass for real images
        real_pair = torch.cat((real_A, real_B), 1)
        pred_real = self.discriminator(real_pair)
        loss_D_real = self.criterionGAN(pred_real, target_is_real=True)

        # Combine discriminator losses and perform backpropagation
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.manual_backward(loss_D)        

        # Update discriminator weights
        opt_D.step()        

        # -------------------- Update Generator --------------------
        # Disable backprop for the discriminator to save computation
        self.set_requires_grad(self.discriminator, False)

        # Zero gradients for the generator
        opt_G.zero_grad()

        # Forward pass for fake images (again) to compute the generator loss
        fake_B = self.forward(real_A)  # you can reuse if no memory issues
        fake_pair = torch.cat((real_A, fake_B), 1)
        pred_fake = self.discriminator(fake_pair)
        loss_G_GAN = self.criterionGAN(pred_fake, target_is_real=True)

        # L1 loss between generated images and real images
        loss_G_L1 = self.criterionL1(fake_B, real_B) * self.lambda_L1

        # Combine generator losses and perform backpropagation
        loss_G = loss_G_GAN + loss_G_L1
        self.manual_backward(loss_G)

        # Update generator weights
        opt_G.step()

        # -------------------- logging and LR schedulers --------------------
                
        train_log_dict = {'train_loss_D': loss_D, 'train_loss_G': loss_G, 'train_loss_G_GAN': loss_G_GAN, 'train_loss_G_L1': loss_G_L1}
        self.log_dict(train_log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # update schedulers
        if self.trainer.is_last_batch: #and (self.trainer.current_epoch + 1) % N == 0:
            sch_D.step()
            sch_G.step()


    def validation_step(self, val_batch, batch_idx):
        real_A, real_B = val_batch
        fake_B = self.generator(real_A)

        val_loss = self.criterionL1(fake_B, real_B) * self.lambda_L1

        # every 50 steps, save 
        if batch_idx % 50 == 0:
            self.plot_validation_example(real_A, real_B, fake_B, batch_idx)        

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return val_loss
    
    def test_step(self, val_batch, batch_idx):
        real_A, real_B = val_batch
        fake_B = self.generator(real_A)

        test_loss = self.criterionL1(fake_B, real_B) * self.lambda_L1

        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return test_loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        # predict
        real_A, real_A_path = batch
        fake_B = self.generator(real_A)

        # Export 
        for img, pth in zip(fake_B, real_A_path):

            # Adjust path for export
            for k, v in self.config['EXPORT']['name_change'].items():
                pth = pth.replace(k, v)
            os.makedirs(os.path.dirname(pth), exist_ok=True)
            pil_img = v2.functional.to_pil_image(img)  # Convert tensor to PIL image
            pil_img.save(pth)  # Save each image        
        return fake_B, real_A_path        


    def configure_optimizers(self):
        # Set up optimizers and schedulers
        optimizer = getattr(torch.optim, self.config['OPTIMIZER']['algorithm'])

        optimizer_g = optimizer(self.generator.parameters(),
                                lr=self.config["OPTIMIZER"]["lr_g"],
                                eps=self.config["OPTIMIZER"]["eps"],
                                betas=(self.config['OPTIMIZER']['beta1_adam'], self.config['OPTIMIZER']['beta2_adam']),
                                weight_decay=self.config['REGULARIZATION']['optimizer_weight_decay'])
        optimizer_d = optimizer(self.discriminator.parameters(),
                                lr=self.config["OPTIMIZER"]["lr_d"],
                                eps=self.config["OPTIMIZER"]["eps"],
                                betas=(self.config['OPTIMIZER']['beta1_adam'], self.config['OPTIMIZER']['beta2_adam']),
                                weight_decay=self.config['REGULARIZATION']['optimizer_weight_decay'])

        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g,
                                                               T_max=self.config['ADVANCEDMODEL']['max_epochs'])
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d,
                                                               T_max=self.config['ADVANCEDMODEL']['max_epochs'])

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


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
        cax3 = axs[1, 0].imshow(abs_diff, cmap='gray', vmin=vmin, vmax=0.3*vmax)
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

    def plot_augmentation(self, raw_proton_images, augmented_proton_images):

        # useful plot to evaluate the impact of augmentations

        B, C, H, W = raw_proton_images.shape

        abs_diff = torch.abs(raw_proton_images - augmented_proton_images)
        for b in range(B):
            rp = raw_proton_images[b, 0, :, :].cpu().numpy()
            ap = augmented_proton_images[b, 0, :, :].cpu().numpy()      
            ad = abs_diff[b, 0, :, :].cpu().numpy()

            vmin, vmax = raw_proton_images.min(), raw_proton_images.max()
            fig, axs = plt.subplots(2, 2, figsize=(15, 6))

            cax1 = axs[0, 0].imshow(rp, cmap='gray', vmin=vmin, vmax=vmax)
            axs[0, 0].set_title('Raw proton image')
            fig.colorbar(cax1, ax=axs[0, 0]) 

            cax2 = axs[0, 1].imshow(ap, cmap='gray', vmin=vmin, vmax=vmax)
            axs[0, 1].set_title('Augmented proton image')
            fig.colorbar(cax2, ax=axs[0, 1])   

            # Third plot (Absolute Difference)
            cax3 = axs[1, 0].imshow(ad, cmap='gray', vmin=vmin, vmax=0.01*vmax)
            axs[1, 0].set_title('Absolute Difference (scale 0.01*max)')
            fig.colorbar(cax3, ax=axs[1, 0])

            # Third plot (Absolute Difference)
            cax4 = axs[1, 1].imshow(ad, cmap='gray', vmin=vmin, vmax=0.1*vmax)
            axs[1, 1].set_title('Absolute Difference (scale 0.1*max)')
            fig.colorbar(cax4, ax=axs[1, 1])  

            plt.tight_layout()    

            save_path = os.path.join(self.config['CHECKPOINT']['logger_folder'], "example_augmentations", f"epoch_{self.current_epoch}")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"image_{b}.png"))

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
# --------------------------------------------------------------------------------------------
class BasicUNetPlusPlus_Monai(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

        self.model = BasicUNetPlusPlus(spatial_dims=2,
                                        in_channels=1,
                                        out_channels=1,
                                        act=self.config['ADVANCEDMODEL']['activation'],
                                        dropout=self.config['ADVANCEDMODEL']['drop_rate'],
                                        features=self.config['ADVANCEDMODEL']['unet_features'])
        init_weights(self.model, init_type = "normal")

    def forward(self, x):
        x = self.model(x)
        return x[0] # not sure why it's in a list, but it's what we get with BasicUnetPlusPlus.


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator. From pix2pix: https://arxiv.org/pdf/1611.07004.pdf"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)          -- the number of channels in input images
            ndf (int)               -- the number of filters in the last conv layer
            n_layers (int)          -- the number of conv layers in the discriminator
            norm_layer              -- normalization layer
            use_spectral_norm (bool) -- if True, apply spectral normalization to conv layers
        """
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1 # padding on all four sides of input.
        
        def apply_spectral_norm(layer):
            return param.spectral_norm(layer) if use_spectral_norm else layer

        sequence = [
            apply_spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                apply_spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            apply_spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            apply_spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))
        ]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)
        init_weights(self.model, init_type="normal")

        # Layer1 with ndf=64 kernel_size=4, ndf=64, stride=2, padding=1
        # [C, H, W] => [ndf, floor((H_in-2)/2 + 1), floor((W_in-2)/2 + 1)]

        # Typical layers:
        # conv1: [C, H, W] -> [ndf, H/2, W/2]
        # conv2: [ndf, H/2, W/2] -> [2*ndf, H/4, W/4]
        # conv3: [2*ndf, H/4, W/4] -> [4*ndf, H/8, W/8]

        # conv4: [4*ndf, H/8, W/8] -> [8*ndf, H/8, W/8]
        # out_conv: [8*ndf, H/8, W/8] -> [1, H/8, W/8].

        # The current config gives a receptive field of ~70x70 pixels. In my opinion it's a bit large.

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, target_real_label=1.0, target_fake_label=0.0, label_smoothing=0, gan_mode="vanilla"):
        """ Initialize the GANLoss class.

        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None

        self.loss = nn.BCEWithLogitsLoss()
        #self.loss = nn.MSELoss()

        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """

        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)

            if self.label_smoothing >0:
                positive_smoothed_labels = 1.0 - self.label_smoothing
                negative_smoothed_labels = self.label_smoothing
                target_tensor = target_tensor * positive_smoothed_labels + \
                    (1 - target_tensor) * negative_smoothed_labels

            loss = self.loss(prediction, target_tensor)

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss        




"""
fake_B = self.generator(real_A)
# Optimise discriminator:  maximize log(D(x,y)) + log(1 - D(x,G(x)))
fake_pair = torch.cat((real_A, fake_B), 1) # used to detach fake_B here.
pred_fake = self.discriminator(fake_pair.detach())
loss_D_fake = self.criterionGAN(pred_fake, target_is_real=False)
real_pair = torch.cat((real_A, real_B), 1)
pred_real = self.discriminator(real_pair)
loss_D_real = self.criterionGAN(pred_real, target_is_real=True)
loss_D = (loss_D_fake + loss_D_real) * 0.5

opt_D.zero_grad()
self.manual_backward(loss_D)
opt_D.step()        

# Optimise generator: maximize log(D(x,G(x))) + L1(y,G(x))
fake_pair = torch.cat((real_A, fake_B), 1)
pred_fake = self.discriminator(fake_pair)
loss_G_GAN = self.criterionGAN(pred_fake, target_is_real=True)  # fake labels are real for generator.
loss_G_L1 = self.criterionL1(fake_B, real_B) * self.lambda_L1 # content loss
loss_G = loss_G_GAN + loss_G_L1

opt_G.zero_grad()
self.manual_backward(loss_G)
opt_G.step()
"""