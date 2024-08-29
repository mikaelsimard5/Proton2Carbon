import ast
import argparse
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import lightning as L
import toml
import torch
import subprocess
import sys

from _Models import UNets, Pix2Pix
from Dataloader.Dataloader_P2C import load_image_dataset, DataModule

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
    gpus_w_lowest_VRAM = [gpu[0] for gpu in gpu_data[:N]]
    print(f"GPUs that will be used: {gpus_w_lowest_VRAM}")
    return gpus_w_lowest_VRAM # get indexes of lowest VRAM use

def load_config(config_file):
    return toml.load(config_file)

def get_transforms(): # transforms made on CPU - should just be image formatting.
    # [!] no normalisation from 0 to 1 is achieved here.
    train_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])
    val_transform = train_transform
    return train_transform, val_transform

def get_logger(config, model_name):
    return L.pytorch.loggers.TensorBoardLogger(config['CHECKPOINT']['logger_folder'], name=model_name)

def get_callbacks(config, model_name):
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=config['CHECKPOINT']['monitor'],
        filename= model_name + '-{epoch:02d}_{val_loss:.2f}',
        save_top_k=1,
        mode=config['CHECKPOINT']['mode'])
    return [lr_monitor, checkpoint_callback]    

def export_config(config, logger):
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(logger.log_dir + "/Config.ini", "w+") as toml_file:        
        toml.dump(config, toml_file) 

if __name__ == "__main__":

    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    config['ADVANCEDMODEL']['n_gpus'] = args.gpus # add to config file to keep on record
    print(f"Available GPUs: {torch.cuda.device_count()}, {config['ADVANCEDMODEL']['n_gpus']} used to train.")

    # Base setup
    L.seed_everything(config['ADVANCEDMODEL']['random_seed'], workers=True)
    torch.set_float32_matmul_precision('medium')        

    # Create logger, callbacks and transforms
    model_name = config['ADVANCEDMODEL']['name']
    logger = get_logger(config, model_name)
    export_config(config, logger)
    callbacks = get_callbacks(config, model_name)
    train_transform, val_transform = get_transforms()

    # Create model
    if config['ADVANCEDMODEL']['name'] == "Pix2Pix":
        model = Pix2Pix.Pix2PixLightning(config)
    elif config['ADVANCEDMODEL']['name'] == "BasicUnetPlusPlus":
        model = UNets.Unet(config)
    else:
        raise ValueError(f"Invalid model name '{config['ADVANCEDMODEL']['name']}'.")

    # Load dataset - this method is specific to the current data organisation.
    image_dataset = load_image_dataset(config)

    #image_dataset = image_dataset.sample(frac=0.1, random_state=42)

    data = DataModule(image_dataset, train_transform=train_transform, val_transform=val_transform, config=config)

    # Create trainer
    devices = N_gpus_with_lowest_VRAM(config['ADVANCEDMODEL']['n_gpus'])
    trainer = L.Trainer(devices=devices,
                        accelerator="gpu",
                        strategy='ddp_find_unused_parameters_true',
                        max_epochs=config['ADVANCEDMODEL']['max_epochs'],
                        precision=config['BASEMODEL']['precision'],
                        callbacks=callbacks,
                        logger=logger,
                        check_val_every_n_epoch=1,
                        log_every_n_steps=5,
                        sync_batchnorm=True)
    
    trainer.fit(model, datamodule=data)
  











