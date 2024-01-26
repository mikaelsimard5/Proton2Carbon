import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch 
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction, target):
        # prediction and target are size [B, C, H, W].

        weights = self.get_weights(target) # size is [B, H, W]

        # uncomment if you want to debug
        # self.save_subplots(target, weights, folder_path="/home/dgs2/Desktop/example_images_P2C/")

        return torch.mean(weights * (prediction - target) ** 2)        

    def get_weights(self, target):

        B, C, H, W = target.shape
        if C != 1:
            raise ValueError(f'Only works with single channel, you have: {C}.')

        PDDs = target.mean(dim=2).squeeze() # [B, W]
        profiles = target.mean(dim=3).squeeze() # [B, H]
        profiles -= torch.quantile(profiles, 0.1, dim=1, keepdim=True)  # remove 10th percentile as simple baseline estimate

        # Get lateral_indexes (imax), depth_indexes (jmax) for the position of the BP.
        depth_indexes = self.PDD_BP_pos_finder_torch(PDDs)

        selected_slices = target[torch.arange(B), :, :, depth_indexes]
        lateral_indexes = torch.argmax(selected_slices, dim=2).squeeze()

        # Get the sigmas for masking.           

        # For sigma_depth_before_peak
        K = 0.1 * depth_indexes.float()
        R = torch.tensor(0.1, device=depth_indexes.device)

        sigma_depth_before_peak = torch.sqrt((K - depth_indexes.float()) ** 2 / (2 * torch.log(1 / R)))
        sigma_depth_after_peak = sigma_depth_before_peak

        # For sigma_lateral
        # Assuming you've computed fwhm_all using find_fwhm_batch()
        fwhm_all = self.find_fwhm_batch(profiles).to(depth_indexes.device)
        K = 3 / 2.355 * fwhm_all.float()
        R = torch.tensor(0.05, device=depth_indexes.device)

        sigma_lateral = torch.sqrt((K - lateral_indexes.float()) ** 2 / (2 * torch.log(1 / R)))    

        weights = self.create_2d_weight_matrix(target.shape,
                                          lateral_indexes,
                                          depth_indexes,
                                          sigma_lateral,
                                          sigma_depth_before_peak,
                                          sigma_depth_after_peak)

        return weights


    def PDD_BP_pos_finder_torch(self, PDDs):
        c = 0.8  # position is defined as c*max
        B, W = PDDs.shape

        # Find jmax for each profile
        jmax_values = torch.argmax(PDDs, dim=1)

        # Calculate maxval for each profile
        maxvals = PDDs[torch.arange(B), jmax_values]

        # Prepare to calculate the position where np.abs(c * maxval - PDD[jmax:]) is minimized
        c_maxvals = c * maxvals.unsqueeze(1)  # Shape [B, 1]
        
        # Using broadcasting to subtract c_maxvals from PDDs for each PDD
        abs_diff = torch.abs(c_maxvals - PDDs)

        # Setting the values before jmax to a large number so they are not considered
        for i in range(B):
            abs_diff[i, :jmax_values[i]] = float('inf')

        # Find the position where abs_diff is minimized for each profile
        positions = torch.argmin(abs_diff, dim=1)# + jmax_values - 1

        return positions

    def find_fwhm_batch(self, profiles):
        B, H = profiles.shape

        # Step 1: Find the maximum value and its index for each profile
        max_vals, max_indices = torch.max(profiles, dim=1)
        half_maxs = max_vals / 2

        fwhms = torch.zeros(B, dtype=torch.int64)  # To store FWHM for each profile
        for i in range(B):
            # Indices before and after the maximum where the vector crosses half max
            indices_before_max = torch.where(profiles[i, :max_indices[i]] < half_maxs[i])[0]
            indices_after_max = torch.where(profiles[i, max_indices[i]:] < half_maxs[i])[0]

            if indices_before_max.size(0) > 0 and indices_after_max.size(0) > 0:
                lower_index = indices_before_max[-1]
                upper_index = max_indices[i] + indices_after_max[0]
                # Step 3: Calculate FWHM
                fwhms[i] = upper_index - lower_index
            else:
                # FWHM could not be determined, set to -1 or any flag value
                fwhms[i] = -1

        return fwhms        


    def create_2d_weight_matrix(self, image_shape, lateral_indexes, depth_indexes, sigma_lateral, sigma_depth_before_peak, sigma_depth_after_peak):
        B, C, H, W = image_shape

        # Create arrays for x and y coordinates
        y_coords = torch.arange(H).view(H, 1).expand(H, W)
        x_coords = torch.arange(W).view(1, W).expand(H, W)

        # Create a 3D grid of coordinates for batch processing
        y_grid = y_coords.unsqueeze(0).expand(B, H, W).to(lateral_indexes.device)
        x_grid = x_coords.unsqueeze(0).expand(B, H, W).to(lateral_indexes.device)

        # Expand lateral_indexes and depth_indexes to 3D.
        # so go from [B] -> [B,1,1] and then expand.
        lateral_indexes_3d = lateral_indexes.unsqueeze(-1).unsqueeze(-1).expand(B, H, W)
        depth_indexes_3d = depth_indexes.unsqueeze(-1).unsqueeze(-1).expand(B, H, W)

        # Expand sigma values to 3D
        sigma_lateral_3d = sigma_lateral.unsqueeze(-1).unsqueeze(-1).expand(B, H, W)
        sigma_depth_before_peak_3d = sigma_depth_before_peak.unsqueeze(-1).unsqueeze(-1).expand(B, H, W)
        sigma_depth_after_peak_3d = sigma_depth_after_peak.unsqueeze(-1).unsqueeze(-1).expand(B, H, W)

        # Calculate 2D weights
        weights1 = torch.exp(-((y_grid - lateral_indexes_3d) ** 2 / (2 * sigma_lateral_3d ** 2)) - ((x_grid - depth_indexes_3d) ** 2 / (2 * sigma_depth_before_peak_3d ** 2)))
        weights2 = torch.exp(-((y_grid - lateral_indexes_3d) ** 2 / (2 * sigma_lateral_3d ** 2)) - ((x_grid - depth_indexes_3d) ** 2 / (2 * sigma_depth_after_peak_3d ** 2)))

        # Combine the two weight matrices
        weights = torch.where(x_grid < depth_indexes_3d, weights1, weights2)

        return weights

    # for debug...
    def save_subplots(self, labels, weights, folder_path="/home/ubuntu/example_images"):
        B = labels.size(0)  # Number of items in the batch

        i = 0 # just for the first element of the batch
        # Convert label and weight tensors to numpy for plotting
        label = labels[i].squeeze().cpu().numpy() # squeeze to remove the channel dim.
        weight = weights[i].cpu().numpy()

        # Create a subplot
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot label
        axs[0].imshow(label, cmap='gray')
        axs[0].set_title('Label')
        axs[0].axis('off')

        # Plot weight map
        axs[1].imshow(weight, cmap='gray')
        axs[1].set_title('Weight Map')
        axs[1].axis('off')

        # Generate a random number for the filename
        rand_num = random.randint(1000, 9999)

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the figure
        plt.savefig(os.path.join(folder_path, f"image_{rand_num}.png"))
        plt.close(fig)        

