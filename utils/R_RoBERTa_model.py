from typing import Any, Optional
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from utils.Score import *
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification
from transformers import AutoModel #추가한것 
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.5, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, target_tensor, weight=self.weight, reduction=self.reduction
        )

class FCLayer(pl.LightningModule):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super().__init__()
        self.save_hyperparameters()

        self.use_activation = use_activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class RRoBERTa(pl.LightningModule):
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

        self.roberta = AutoModel.from_pretrained(MODEL_NAME)

        self.cls_fc_layer = FCLayer(self.model_config.hidden_size, self.model_config.hidden_size//2, 0)
        self.entity_fc_layer = FCLayer(self.model_config.hidden_size, self.model_config.hidden_size//2, 0)
        self.label_classifier = FCLayer(
            self.model_config.hidden_size//2 * 3,
            self.model_config.num_labels,
            0,
            use_activation=False,
        )
        
        self.loss_dict = {
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
            'FocalLoss': FocalLoss()
            }
        self.loss_func = self.loss_dict[loss]

    def forward(self, x):
        """
        model gets x and then predicts probs of each category through pre-selected model, LSTM, and classfier(fc) Layer
        """
        output = self.roberta(input_ids=x['input_ids'],attention_mask=x['attention_mask'], token_type_ids=x['token_type_ids'])[0]
 
        sentence_end_position = torch.where(x["input_ids"] == 2)[1]
        sent1_end, sent2_end = sentence_end_position[0], sentence_end_position[1]

        cls_vector = output[:,0,:]
        subject_vector = output[:,1:sent1_end]
        object_vector = output[:,sent1_end+1:sent2_end]

        subject_vector = torch.mean(subject_vector, dim=1)
        object_vector = torch.mean(object_vector, dim=1)

        cls_embedding = self.cls_fc_layer(cls_vector)
        subject_embedding = self.entity_fc_layer(subject_vector)
        object_embedding = self.entity_fc_layer(object_vector)

        concat_embedding = torch.cat([cls_embedding, subject_embedding, object_embedding], dim=-1)
        return self.label_classifier(concat_embedding)
    
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