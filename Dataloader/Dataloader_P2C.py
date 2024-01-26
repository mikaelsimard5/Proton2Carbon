import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from PIL import Image

def load_image_dataset(config):

    image_dataset = []

    proton_dirs = [os.path.join(config['DATA']['Base_Folder'], "protons", dataset, view, 'corrected_cropped')
                    for dataset in config['DATA']['Datasets'] for view in config['DATA']['Views']]
    datasets    = [dataset for dataset in config['DATA']['Datasets'] for view in config['DATA']['Views']]            

    if not config['ADVANCEDMODEL']['Inference']:
            carbon_dirs = [os.path.join(config['DATA']['Base_Folder'], "carbon", dataset, view, 'corrected_cropped') 
                           for dataset in config['DATA']['Datasets'] for view in config['DATA']['Views']]                
    else:
        carbon_dirs = ['NA'] * len(proton_dirs)  # if inference, no carbon path needed

    # Iterate over all directories and collect image pairs
    for carbon_dir, proton_dir, dataset in zip(carbon_dirs, proton_dirs, datasets):
        carbon_images = sorted(os.listdir(carbon_dir))
        proton_images = sorted(os.listdir(proton_dir))

        for carbon_img_name, proton_img_name in zip(carbon_images, proton_images):
            carbon_img_path = os.path.join(carbon_dir, carbon_img_name)
            proton_img_path = os.path.join(proton_dir, proton_img_name)
            image_dataset.append({'dataset': dataset, 'carbon_image_path': carbon_img_path, 'proton_image_path': proton_img_path})

    image_dataset = pd.DataFrame(image_dataset)        
    return image_dataset


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, image_dataset, config=None, transform=None):
        
        super().__init__()
        self.image_dataset = image_dataset
        self.inference = config['ADVANCEDMODEL']['Inference']
        self.transform = transform
        self.image_save_folder = config['CHECKPOINT']['logger_folder']
    
    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):

        row = self.image_dataset.iloc[idx]
        proton_image = Image.open(row['proton_image_path']).convert('L')
        proton_image = self.transform(proton_image)

        if self.inference:
            return proton_image, None
        else:
            carbon_image = Image.open(row['carbon_image_path']).convert('L')
            carbon_image = self.transform(carbon_image)

            if idx % 1000 == 0:

                # Save every 1000th image
                c_img = np.array(255.0 * carbon_image, dtype='uint8')
                p_img = np.array(255.0 * proton_image, dtype='uint8')

                c_img = Image.fromarray(c_img[0, :, :], "L").convert("RGB")
                p_img = Image.fromarray(p_img[0, :, :], "L").convert("RGB")

                # Recolor the images (e.g., one red, one green)
                red_image = Image.merge('RGB', (c_img.split()[0], Image.new('L', c_img.size), Image.new('L', c_img.size)))
                green_image = Image.merge('RGB', (Image.new('L', p_img.size), p_img.split()[0], Image.new('L', p_img.size)))

                # Blend the images
                bp = os.path.join(self.image_save_folder, "example_images", "raw_data")
                os.makedirs(bp, exist_ok=True)
                blended_image = Image.blend(red_image, green_image, alpha=0.5)
                blended_image.save(os.path.join(bp, f"example_blend_{idx}.png"))                

            return proton_image, carbon_image


class DataModule(LightningDataModule):
    def __init__(self, image_dataset, train_transform=None, val_transform=None, config=None, **kwargs):
        super().__init__()

        self.batch_size = config['BASEMODEL']['Batch_Size']

        if isinstance(config['ADVANCEDMODEL']['n_gpus'], list):
            n_gpus = len(config['ADVANCEDMODEL']['n_gpus'])
        else:
            n_gpus = config['ADVANCEDMODEL']['n_gpus']
        self.num_workers = int(.8 * mp.Pool()._processes / n_gpus)

        # Sampling of image_dataset by dataset.
        unique_datasets = image_dataset['dataset'].unique()

        # Calculate split sizes
        train_size = int(len(unique_datasets) * config['DATA']['Train_Size'])
        val_size = len(unique_datasets) - train_size     

        # Randomly sample dataset names for training and validation sets
        train_datasets = set(np.random.choice(unique_datasets, train_size, replace=False))
        val_datasets = set(unique_datasets) - train_datasets 

        print(f"Datasets in the training set: {train_datasets}.")
        print(f"Datasets in the validation set: {val_datasets}.")
        
        # Split the original DataFrame
        train_image_dataset = image_dataset[image_dataset['dataset'].isin(train_datasets)]
        val_image_dataset = image_dataset[image_dataset['dataset'].isin(val_datasets)]   

        train_image_dataset = train_image_dataset.reset_index(drop=True)
        val_image_dataset = val_image_dataset.reset_index(drop=True)

        print(f'There are {len(train_image_dataset)} images in the training dataset, and {len(val_image_dataset)} images in the testing dataset.')

        self.train_data = DataGenerator(train_image_dataset, config=config, transform=train_transform, **kwargs)
        self.val_data = DataGenerator(val_image_dataset, config=config, transform=val_transform, **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
                          shuffle=False)    



