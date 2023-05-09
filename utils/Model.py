from typing import Any, Optional
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from utils.Score import *
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification


class Model(pl.LightningModule):
    def __init__(self, MODEL_NAME, model_config, lr, loss, optim, scheduler):
        '''
        모델 생성
        MODEL_NAME: 사용할 모델
        model_config: 사용할 모델 config
        lr: 모델의 lr
        '''
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.test_step_outputs = []

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
        model gets x->predict probs of each category
        """
        x = self.classifier(x['input_ids'],x['attention_mask'],x['token_type_ids'])

        return x['logits']
    
    def training_step(self, batch, batch_idx):
        """
        calc train score & loss
        """
        logits = self(batch)
        y = batch['labels']

        loss = self.loss_func(logits, y)
        self.log("train_loss: ",loss)

        logits = logits.detach().cpu()
        y = y.detach().cpu()
        preds = np.argmax(logits.numpy(), axis=-1)

        f1 = klue_re_micro_f1(preds, y)
        acc = accuracy_score(y, preds)
        self.log("train_f1_score: ", f1)
        self.log("train_acc_score: ", acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        calc valid score
        """
        logits = self(batch)
        y = batch['labels']

        loss = self.loss_func(logits, y)
        self.log("valid_loss: ",loss)

        y = y.detach().cpu()
        logits = logits.detach().cpu()
        preds = np.argmax(logits.numpy(), axis=-1)

        f1 = klue_re_micro_f1(preds, y)
        acc = accuracy_score(y, preds)
        self.log("valid_f1_score: ", f1)
        self.log("valid_acc_score: ", acc)

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
        self.log("test_f1_score: ", f1)
        self.log("test_acc_score: ", acc)
        
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
        self.optimizer_dict={
            'AdamW': torch.optim.AdamW(self.parameters(), lr=self.lr)
            }
        self.lr_scheduler_dict={
        }
        """
        use AdamW as optimizer
        """
        optimizer = self.optimizer_dict[self.optim]
        if self.scheduler == 'None':
            return optimizer
        else:
            scheduler = self.lr_scheduler_dict[self.scheduler]
            return [optimizer], [scheduler]