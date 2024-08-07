import ast
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import lightning as L
import toml
import torch
import sys

from _Models import UNets, Pix2Pix
from Dataloader.Dataloader_P2C import load_image_dataset, DataModule

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
        monitor=config['CHECKPOINT']['Monitor'],
        filename= model_name + '-{epoch:02d}_{val_loss:.2f}',
        save_top_k=1,
        mode=config['CHECKPOINT']['Mode'])
    return [lr_monitor, checkpoint_callback]    

def export_config(config, logger):
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(logger.log_dir + "/Config.ini", "w+") as toml_file:        
        toml.dump(config, toml_file) 

if __name__ == "__main__":

    # Load configuration
    config = load_config(sys.argv[1])

    config['ADVANCEDMODEL']['n_gpus'] = config['ADVANCEDMODEL']['n_gpus']
    print(f"Available GPUs: {torch.cuda.device_count()}, {config['ADVANCEDMODEL']['n_gpus']} used to train.")

    # Base setup
    L.seed_everything(config['ADVANCEDMODEL']['Random_Seed'], workers=True)
    torch.set_float32_matmul_precision('medium')        

    # Create logger, callbacks and transforms
    model_name = config['ADVANCEDMODEL']['Name']
    logger = get_logger(config, model_name)
    export_config(config, logger)
    callbacks = get_callbacks(config, model_name)
    train_transform, val_transform = get_transforms()

    # Create model
    if config['ADVANCEDMODEL']['Name'] == "Pix2Pix":
        model = Pix2Pix.Pix2PixLightning(config)
    elif config['ADVANCEDMODEL']['Name'] == "BasicUnetPlusPlus":
        model = UNets.Unet(config)
    else:
        raise ValueError(f"Invalid model name '{config['ADVANCEDMODEL']['Name']}'.")

    # Create dataset
    image_dataset = load_image_dataset(config)
    #image_dataset = image_dataset[:70000]

    data = DataModule(image_dataset, train_transform=train_transform, val_transform=val_transform, config=config)

    # Create trainer
    trainer = L.Trainer(devices=config['ADVANCEDMODEL']['n_gpus'],
                        accelerator="gpu",
                        strategy='ddp_find_unused_parameters_true',
                        max_epochs=config['ADVANCEDMODEL']['Max_Epochs'],
                        precision=config['BASEMODEL']['Precision'],
                        callbacks=callbacks,
                        logger=logger,
                        check_val_every_n_epoch=1,
                        log_every_n_steps=5,
                        sync_batchnorm=True)
    
    trainer.fit(model, datamodule=data)
  











