import ast
import argparse
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import lightning as L
import multiprocessing as mp
import pandas as pd
import toml
import torch
import subprocess
import sys

from _Models import UNets, Pix2Pix
from Dataloader.Dataloader_P2C import DataModule, DataGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Proton2Carbon trainer')
    parser.add_argument('--config', type=str, default=None, help='path of the configuration file')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus to use for training.')
    args = parser.parse_args()
    return args

def N_gpus_with_lowest_VRAM(N=2):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    gpu_data = [] # Parse the output into a list of tuples (index, memory_used)
    for line in output.strip().split('\n'):
        index, memory_used = line.split(', ')
        gpu_data.append((int(index), int(memory_used)))
    gpu_data.sort(key=lambda x: x[1]) # Sort the GPUs by memory used (ascending order)
    return [gpu[0] for gpu in gpu_data[:N]] # get indexes of lowest VRAM use

def load_config(config_file):
    return toml.load(config_file)

def get_transforms(): # transforms made on CPU - should just be image formatting.
    # [!] no normalisation from 0 to 1 is achieved here.
    train_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
    val_transform = train_transform
    return train_transform, val_transform

def get_logger(config, model_name):
    return L.pytorch.loggers.TensorBoardLogger(config['CHECKPOINT']['logger_folder'], name=model_name)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def export_config(config, logger):
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(logger.log_dir + "/Config.ini", "w+") as toml_file:        
        toml.dump(config, toml_file) 

if __name__ == "__main__":

    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    config['ADVANCEDMODEL']['n_gpus'] = args.gpus # add to config file to keep on record
    num_workers = int(.8 * mp.Pool()._processes / config['ADVANCEDMODEL']['n_gpus'])
    print(f"Available GPUs: {torch.cuda.device_count()}, {config['ADVANCEDMODEL']['n_gpus']} used to train.")

    # Base setup
    torch.set_float32_matmul_precision('medium')        

    # Create logger, callbacks and transforms
    train_transform, val_transform = get_transforms()
    
    # Create model
    if config['ADVANCEDMODEL']['name'] == "Pix2Pix":
        model = Pix2Pix.Pix2PixLightning.load_from_checkpoint(config['CHECKPOINT']['name'])
        train_config = Pix2Pix.Pix2PixLightning.read_config_from_checkpoint(config['CHECKPOINT']['name'])
    elif config['ADVANCEDMODEL']['name'] == "BasicUnetPlusPlus":
        model = UNets.Unet.load_from_checkpoint(config['CHECKPOINT']['name'])
        train_config = UNets.Unet.read_config_from_checkpoint(config['CHECKPOINT']['name'])
    else:
        raise ValueError(f"Invalid model name '{config['ADVANCEDMODEL']['name']}'.")
    
    # Set model to evaluation/inference mode
    model.config['ADVANCEDMODEL']['inference'] = True
    model.config['EXPORT'] = config['EXPORT']
    model.eval()
    model = freeze_model(model)

    # Start the trainer
    trainer = L.Trainer(devices=N_gpus_with_lowest_VRAM(config['ADVANCEDMODEL']['n_gpus']),
                        accelerator="gpu",
                        strategy="ddp_find_unused_parameters_true",
                        logger=False,
                        precision=train_config['BASEMODEL']['precision'], # should now match the training model
                        use_distributed_sampler = False,
                        benchmark=False)    

    # Load dataset - puts in memory the path of all 
    images_paths = []
    for pdir in config['DATA']['infer_folders']:
        files = [os.path.join(pdir, file) for file in sorted(os.listdir(pdir))]
        images_paths.extend(files)
    image_dataset = pd.DataFrame({'proton_image_path':images_paths})
    #image_dataset = image_dataset[0:510]

    # Multi-GPU spport - manually sample and infer parts of the dataset
    images_per_gpu = len(image_dataset) // trainer.world_size
    start_idx     = trainer.global_rank * images_per_gpu
    end_idx = start_idx + images_per_gpu if trainer.global_rank < trainer.world_size - 1 else len(image_dataset)    
    print(f"trainer with rank {trainer.global_rank} uses indexes {start_idx} to {end_idx} for tile_dataset originally of length { len(image_dataset)}.")
    image_dataset = image_dataset[start_idx:end_idx]

    # Get the data
    num_workers_dataloader = int(.8 * mp.Pool()._processes / config['ADVANCEDMODEL']['n_gpus'])
    val_data = DataGenerator(image_dataset, config=config, transform=val_transform)
    data = DataLoader(val_data, batch_size=config['BASEMODEL']['batch_size'],
                      num_workers= num_workers_dataloader, pin_memory=False, shuffle=False)        

    # Predict - predicions are exported as png directly from the predict() method
    predictions = trainer.predict(model, data)










