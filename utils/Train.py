import pickle as pickle
import torch
import pytorch_lightning as pl
from transformers import AutoConfig
from utils.Model import Model
from utils.DataLoader import DataLoader
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from utils.customModel import  customModel


def train(cfg):
    '''
    모델 설정은 기본 설정을 그대로 가져오고 사용하는 레이블의 개수만 현재 데이터에 맞춰서 설정
    '''
    save_path, folder_name = cfg['save_path'], cfg['folder_name']
    model_config = AutoConfig.from_pretrained(cfg['model']['model_name'])
    model_config.num_labels = 30
    """
    model = Model(cfg['model']['model_name'],
                  model_config,cfg['model']['LR'], 
                  cfg['model']['LossF'], 
                  cfg['model']['optim'], 
                  cfg['model']['scheduler'])
    """
    #기존Model+biLSTM
    model = customModel(cfg['model']['model_name'],
                  model_config,cfg['model']['LR'], 
                  cfg['model']['LossF'], 
                  cfg['model']['optim'], 
                  cfg['model']['scheduler'])
    
    # logger 생성
    '''
    pip install wandb
    첫 실행할 때 로그인을 위해 본인 api key를 복사해서 붙여넣어주세요
    wandblog는 results 폴더에 실행모델과 함께 저장됩니다
    '''
    wandb.init(name=folder_name, project="KLUE", entity="hypesalmon", dir=save_path)
    wandb_logger = WandbLogger(save_dir=save_path)
    wandb_logger.experiment.config.update(cfg)

    early_stopping = EarlyStopping(
        monitor = cfg['EarlyStopping']['monitor'],
        min_delta=cfg['EarlyStopping']['min_delta'],
        patience=cfg['EarlyStopping']['patience'],
        verbose=cfg['EarlyStopping']['verbose'],
        mode='max',
    )

    checkpoint = ModelCheckpoint(
        dirpath ='./checkpoints/',
        filename = cfg['model']['model_name']+'-{epoch}-{valid_f1_score:.2f}-{valid_acc_score:.2f}',
        every_n_epochs = 1
    )
    
    trainer = pl.Trainer(accelerator = "auto",
                         max_epochs = cfg['model']['epoch'],
                         log_every_n_steps = 1,
                         logger = wandb_logger,
                         callbacks=[early_stopping, checkpoint] if cfg['EarlyStopping']['turn_on'] else [checkpoint])
    
    dataloader = DataLoader(cfg['model']['model_name'], cfg['model']['batch_size'], cfg['model']['max_len'], cfg['model']['shuffle'])
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    torch.save(model, f'{save_path}/{folder_name}_model.pt')