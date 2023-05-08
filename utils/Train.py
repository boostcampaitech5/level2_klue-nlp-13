import pickle as pickle
import torch
import pytorch_lightning as pl
from transformers import AutoConfig
from utils.Model import Model
from utils.DataLoader import DataLoader


def train(cfg):
    '''모델 설정은 기본 설정을 그대로 가져오고 사용하는 레이블의 개수만 현재 데이터에 맞춰서 설정'''
    model_config = AutoConfig.from_pretrained(cfg['train']['model'])
    model_config.num_labels = 30
    model = Model(cfg['train']['model'],model_config,cfg['train']['LR'], cfg['train']['LossF'], cfg['train']['optim'], cfg['train']['scheduler'])
    
    trainer = pl.Trainer(accelerator = "auto",max_epochs = cfg['train']['epoch'],log_every_n_steps = 1)
    dataloader = DataLoader(cfg['train']['model'], cfg['train']['batch_size'], cfg['train']['shuffle'])
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    save_path, folder_name = cfg['save_path'], cfg['folder_name']
    torch.save(model, f'{save_path}/{folder_name}_model.pt')