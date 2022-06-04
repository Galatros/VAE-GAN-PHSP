import yaml
import os

from models import *
# from helpers import model_architectures
from helpers.PhotonsDataModule import PhotonsDataModule
from helpers.yaml_helper import MyDumper
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers



if __name__ == '__main__':
    # Hyperparameters
    CONFIG_PATH = '/home/jakmic/Projekty/dose3d-phsp/AE_VAE/Lightning_Autoencoders/configs/InfoVAE.yaml'
    LOAD_CHECKPOINT_PATH = None


    # Load config
    with open(CONFIG_PATH, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Prepare logger
    csv_logger = pl_loggers.CSVLogger(save_dir=config['CSVLogger_params']['save_dir'], name=config['CSVLogger_params']['name']) #, version=config['CSVLogger_params']['version']

    # Prepare DataLoaders
    dm=PhotonsDataModule(**config['DataModule_params'])

    # Prepare model
    model = vae_models[config['model_params']['name']](**config['model_params'])

    # Prepare model new version
    # model_architecture=model_architectures[config['model_params']['model_architecture_name']]()
    # model = vae_models[config['model_params']['name']](model_architecture, **config['model_params'])

    # Prepare checkpointing options
    save_checkpoints_path=os.path.join(csv_logger.log_dir,'checkpoints')
    print(csv_logger.log_dir)
    checkpoint_callback = ModelCheckpoint(dirpath=save_checkpoints_path,**config['checkpoint_params'])


    # Prepare trainer and train the model depending on whether it is loaded from checkpoint or not
    if LOAD_CHECKPOINT_PATH is not None:
        # Prepare trainer and trian model from checkpoint
        trainer = Trainer(gpus=1, logger=csv_logger, callbacks=[checkpoint_callback],**config['trainer_params'])
        trainer.fit(model, dm, ckpt_path=LOAD_CHECKPOINT_PATH)
        tmp_dict={"loaded_checkpoint_info":{"trained_from_checkpoint": True, 'loaded_checkpoint_path': LOAD_CHECKPOINT_PATH}}
        config.update(tmp_dict)
    else:
        # Prepare trainer and trian model
        trainer = Trainer(gpus=1, logger=csv_logger, callbacks=[checkpoint_callback],**config['trainer_params'])
        trainer.fit(model, dm)
    
    # Add version name to config
    #config['CSVLogger_params']['version']=csv_logger.version

    # Save the configuration file in the location used by the logger
    save_config_path=os.path.join(csv_logger.log_dir,'config.yaml')
    with open(save_config_path, 'w') as file:
        documents = yaml.dump(config, file, Dumper=MyDumper, sort_keys=False)