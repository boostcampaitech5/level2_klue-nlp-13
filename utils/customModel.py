from typing import Any, Optional
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from utils.Score import *
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel,AutoConfig
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

class customModel(pl.LightningModule):
    def __init__(self, MODEL_NAME, model_config, lr, loss, optim, scheduler,max_len):
        '''
        custom 모델 생성
        기존에 선택한 model에 biLSTM을 붙인 모델

        MODEL_NAME: 사용할 모델
        model_config: 사용할 모델 config
        lr: 모델의 lr
        '''
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.test_step_outputs = []

        #custom된 부분 
        self.num_classes=30
        self.customModel = AutoModel.from_pretrained(MODEL_NAME)
        self.hidden_size = max_len
        self.lstm = nn.LSTM(1024,hidden_size=self.hidden_size,batch_first=True,bidirectional=True,dropout=0.2)
        self.fc = nn.Linear(self.hidden_size*2,self.num_classes)

        self.MODEL_NAME = MODEL_NAME
        self.model_config = model_config
        self.lr = lr
        self.optim = optim        
        self.scheduler = scheduler
        
        self.classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        self.loss_dict = {
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
            }
        self.loss_func = self.loss_dict[loss]

    def forward(self, x):
        """
        model gets x and then predicts probs of each category through pre-selected model, LSTM, and classfier(fc) Layer
        """
        outputs = self.customModel(input_ids=x['input_ids'],attention_mask=x['attention_mask'],token_type_ids=x['token_type_ids'])
        lstm_output,(last_hidden,last_cell) = self.lstm(outputs[0])
        hidden = torch.cat((last_hidden[0],last_hidden[1]),dim=1)
        logits = self.fc(hidden)

        return logits
    
    def training_step(self, batch, batch_idx):
        """
        calc train score & loss
        """
        logits = self(batch)
        y = batch['labels']

        loss = self.loss_func(logits, y)
        self.log("train_loss",loss)

        logits = logits.detach().cpu()
        y = y.detach().cpu()
        preds = np.argmax(logits.numpy(), axis=-1)

        f1 = klue_re_micro_f1(preds, y)
        acc = accuracy_score(y, preds)
        self.log("train_f1_score", f1)
        self.log("train_acc_score", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        calc valid score
        """
        logits = self(batch)
        y = batch['labels']

        loss = self.loss_func(logits, y)
        self.log("valid_loss",loss)

        y = y.detach().cpu()
        logits = logits.detach().cpu()
        preds = np.argmax(logits.numpy(), axis=-1)

        f1 = klue_re_micro_f1(preds, y)
        acc = accuracy_score(y, preds)
        self.log("valid_f1_score", f1)
        self.log("valid_acc_score", acc)

        self.validation_step_outputs.append({"logits": logits, "y": y})
    
    def on_validation_epoch_end(self):
        """
        calc auprc score on validation data
        """
        outputs = self.validation_step_outputs

        logits = torch.cat([x['logits'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])

        logits = logits.detach().cpu().numpy()
        y = y.detach().cpu()

        auprc = klue_re_auprc(logits, y)
        self.log("val_auprc", auprc)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        '''
        calc test score
        '''
        logits = self(batch)
        y = batch['labels']

        logits = logits.detach().cpu()
        y = y.detach().cpu()
        preds = np.argmax(logits.numpy(), axis=-1)

        f1 = klue_re_micro_f1(preds, y)
        acc = accuracy_score(y, preds)
        self.log("test_f1_score", f1)
        self.log("test_acc_score", acc)
        
        self.test_step_outputs.append({"logits": logits, "y": y})

    def on_test_epoch_end(self):
        '''
        calc auprc score on test data
        '''
        outputs = self.test_step_outputs

        logits = torch.cat([x['logits'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])

        logits = logits.detach().cpu().numpy()
        y = y.detach().cpu()

        auprc = klue_re_auprc(logits, y)
        self.log("test_auprc", auprc)
        self.test_step_outputs.clear()

    def predict_step(self, batch: Any, batch_idx):
        '''
        model gets test data->predict probs of each category
        '''
        logits = self(batch)
        return logits
    
    def configure_optimizers(self):
        """
        use AdamW as optimizer and use StepLR as scheduler
        """
        self.optimizer_dict={
            'AdamW': torch.optim.AdamW(self.parameters(), lr=self.lr)
            }
        optimizer = self.optimizer_dict[self.optim]
        self.lr_scheduler_dict={
            'StepLR': StepLR(optimizer, step_size=1, gamma = 0.5)
        }
        if self.scheduler == 'None':
            return optimizer
        else:
            scheduler = self.lr_scheduler_dict[self.scheduler]
            return [optimizer], [scheduler]