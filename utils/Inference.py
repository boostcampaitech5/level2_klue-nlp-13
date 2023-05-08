import os
import pandas as pd
import pickle as pickle
import pytorch_lightning as pl
import torch
from utils.Model import Model
from utils.DataLoader import DataLoader
from utils.Utils import *



def inference(cfg):
    save_path, folder_name = cfg['save_path'], cfg['folder_name']
    dataloader = DataLoader(cfg['inference']['model'], cfg['inference']['batch_size'])

    trainer = pl.Trainer(accelerator="auto")

    model = torch.load(f'{save_path}/{folder_name}_model.pt')
    predicts = trainer.predict(model, datamodule=dataloader)

    pred, prob = get_result(predicts)
    pred = num_to_label(pred)

    output = pd.DataFrame({'id':[i for i in range(len(pred))],'pred_label':pred,'probs':prob})
    print(output)
    output.to_csv(f'{save_path}/{folder_name}_predict.csv', index=False)