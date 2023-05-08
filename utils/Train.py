import pickle as pickle
import torch
import pytorch_lightning as pl
from transformers import AutoConfig
from utils.Model import Model
from utils.DataLoader import DataLoader
from pytorch_lightning.loggers import WandbLogger
import wandb


def train(cfg):
    '''모델 설정은 기본 설정을 그대로 가져오고 사용하는 레이블의 개수만 현재 데이터에 맞춰서 설정'''
    save_path, folder_name = cfg['save_path'], cfg['folder_name']
    model_config = AutoConfig.from_pretrained(cfg['train']['model'])
    model_config.num_labels = 30
    model = Model(cfg['train']['model'],
                  model_config,cfg['train']['LR'], 
                  cfg['train']['LossF'], 
                  cfg['train']['optim'], 
                  cfg['train']['scheduler'])

    # logger 생성
    '''
    pip install wandb
    첫 실행할 때 로그인을 위해 본인 api key를 복사해서 붙여넣어주세요
    wandblog는 results 폴더에 실행모델과 함께 저장됩니다
    '''
    wandb.init(name=folder_name, project="KLUE", entity="hypesalmon", dir=save_path)
    wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(cfg)
    
    trainer = pl.Trainer(accelerator = "auto",
                         max_epochs = cfg['train']['epoch'],
                         log_every_n_steps = 1,
                         logger = wandb_logger)
    
    dataloader = DataLoader(cfg['train']['model'], cfg['train']['batch_size'], cfg['train']['shuffle'])
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    torch.save(model, f'{save_path}/{folder_name}_model.pt')