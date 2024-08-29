import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from PIL import Image

def load_image_dataset(config): # one can create another custom load_image_dataset for other data organisations.

    image_dataset = []
    base_folder = config['DATA']['base_folder']
    datasets = config['DATA']['train_datasets'] + config['DATA']['val_datasets'] + config['DATA']['test_datasets']
    views = config['DATA']['views']    

    proton_dirs = [os.path.join(base_folder, dataset, "protons", view)
                    for dataset in datasets for view in views]            

    if not config['ADVANCEDMODEL']['inference']:
            carbon_dirs = [os.path.join(base_folder, dataset, "carbon", view) 
                           for dataset in datasets for view in views]                
    else:
        carbon_dirs = ['NA'] * len(proton_dirs)  # if inference, no carbon path needed

    repeated_datasets    = [dataset for dataset in datasets for _ in views] #so they fit with views

    # Iterate over all directories and collect image pairs
    for carbon_dir, proton_dir, dataset in zip(carbon_dirs, proton_dirs, repeated_datasets):
        carbon_images = sorted(os.listdir(carbon_dir))
        proton_images = sorted(os.listdir(proton_dir))

        for carbon_img_name, proton_img_name in zip(carbon_images, proton_images):
            carbon_img_path = os.path.join(carbon_dir, carbon_img_name)
            proton_img_path = os.path.join(proton_dir, proton_img_name)
            image_dataset.append({'dataset': dataset, 'carbon_image_path': carbon_img_path, 'proton_image_path': proton_img_path})

    image_dataset = pd.DataFrame(image_dataset)        
    return image_dataset


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, image_dataset, config=None, transform=None, export_examples=False):
        
        super().__init__()
        self.image_dataset = image_dataset
        self.inference = config['ADVANCEDMODEL']['inference']
        self.transform = transform
        self.image_save_folder = config['CHECKPOINT'].get('logger_folder', './')
        self.export_examples = export_examples
    
    def __len__(self):
        return len(self.image_dataset)
    
    def export_examples(self, proton_image, carbon_image, idx):
            # Export examples to visualise results as training occurs.
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

    def __getitem__(self, idx):

        row = self.image_dataset.iloc[idx]
        proton_image = Image.open(row['proton_image_path']).convert('L')
        proton_image = self.transform(proton_image)

        if self.inference:
            return proton_image, row['proton_image_path']
        else:
            carbon_image = Image.open(row['carbon_image_path']).convert('L')
            carbon_image = self.transform(carbon_image)

            # Export training examples & predictions during training
            if (idx%1000==0) & self.export_examples:
                self.save_examples(carbon_image, proton_image, idx)                
            return proton_image, carbon_image


class DataModule(LightningDataModule):
    def __init__(self, image_dataset, train_transform=None, val_transform=None, config=None, **kwargs):
        super().__init__()

        self.batch_size = config['BASEMODEL']['batch_size']
        self.num_workers = int(.8 * mp.Pool()._processes / config['ADVANCEDMODEL']['n_gpus'])

        train_datasets = config['DATA']['train_datasets']
        val_datasets = config['DATA']['val_datasets']
        test_datasets = config['DATA']['test_datasets']
        
        # Split the original DataFrame
        train_image_dataset = image_dataset[image_dataset['dataset'].isin(train_datasets)].reset_index(drop=True)
        val_image_dataset = image_dataset[image_dataset['dataset'].isin(val_datasets)].reset_index(drop=True)

        print(f'There are {len(train_image_dataset)} images in the training dataset, and {len(val_image_dataset)} images in the testing dataset.')

        self.train_data = DataGenerator(train_image_dataset, config=config, transform=train_transform, **kwargs)
        self.val_data = DataGenerator(val_image_dataset, config=config, transform=val_transform, **kwargs)

        if len(test_datasets):
            test_image_dataset = image_dataset[image_dataset['dataset'].isin(test_datasets)].reset_index(drop=True)
            self.test_data = DataGenerator(test_image_dataset, config=config, transform=val_transform, **kwargs)
        else:
            self.test_data = None

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False,
                          shuffle=False)