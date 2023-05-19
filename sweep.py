import yaml
import os
import pickle as pickle
import torch
import pytorch_lightning as pl
import wandb
from utils.Inference import *
from transformers import AutoConfig
from utils.Model import Model
from utils.DataLoader import DataLoader
from utils.Utils import get_folder_name
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
    get_folder_name(cfg)
    save_path, folder_name = cfg['save_path'], cfg['folder_name']

    # config의 sweep부분을 불러옵니다.
    sweep_config = cfg['sweep']
    ver = 0

    # set version to save model

    def train():
        '''
        기존의 train 함수와 다른 것은 없으나, sweep으로 조정하고 싶은 변수는
        cfg['model']['parameter'] 형식에서 config.parameter 로 바꾸어 입력해주세요!
        '''
        global ver
        cfg['version'] = ver
            # logger 생성
        '''
        pip install wandb
        첫 실행할 때 로그인을 위해 본인 api key를 복사해서 붙여넣어주세요
        wandblog는 results 폴더에 실행모델과 함께 저장됩니다
        '''
        wandb.init(name=folder_name, project="KLUE", entity="Hype연어", dir=save_path)
        wandb_logger = WandbLogger(save_dir=save_path)
        wandb_logger.experiment.config.update(cfg)

        config = wandb.config
        '''모델 설정은 기본 설정을 그대로 가져오고 사용하는 레이블의 개수만 현재 데이터에 맞춰서 설정'''
        model_config = AutoConfig.from_pretrained(cfg['model']['model_name'])
        model_config.num_labels = 30
        model = Model(cfg['model']['model_name'],
                    model_config,
                    config.lr, 
                    cfg['model']['LossF'], 
                    cfg['model']['optim'], 
                    cfg['model']['scheduler'])


        early_stopping = EarlyStopping(
            monitor = cfg['EarlyStopping']['monitor'],
            min_delta=cfg['EarlyStopping']['min_delta'],
            patience=cfg['EarlyStopping']['patience'],
            verbose=cfg['EarlyStopping']['verbose'],
            mode='max',
        )

        checkpoint = ModelCheckpoint(
            monitor='valid_loss',
            mode = 'min',
            save_top_k=1,
            save_last=True,
            save_weights_only=True,
            verbose=False,
            dirpath =f'{save_path}/ver{ver}_ckpt',
            filename = cfg['model']['model_name']+'-{epoch}-{valid_f1_score:.2f}-{valid_acc_score:.2f}'
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(accelerator = "auto",
                            max_epochs = cfg['model']['epoch'],
                            log_every_n_steps = 1,
                            logger = wandb_logger,
                            callbacks=[early_stopping, checkpoint,lr_monitor] if cfg['EarlyStopping']['turn_on'] else [checkpoint],
                            precision=16) #fp16 사용
        
        #dataloader = DataLoader(cfg['model']['model_name'], config.batch_size, config.max_len, cfg['model']['shuffle'])
        dataloader = DataLoader(cfg['model']['model_name'], cfg['model']['batch_size'], cfg['model']['max_len'], cfg['model']['shuffle'])
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        torch.save(model, f'{save_path}/{folder_name}_ver{ver}_model.pt')
        ver += 1
        inference(cfg)


    sweep_id = wandb.sweep(sweep = sweep_config, project = 'Sweeps')
    wandb.agent(sweep_id=sweep_id, function=train, count=cfg['sweepcnt'])
    
