import pandas as pd
import pickle as pickle
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer
from utils.Dataset import Dataset
from utils.Utils import label_to_num
from utils.DataPreprocessing import remove_duplicate, use_sotype_token

class DataLoader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle=True):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = 8
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def load_data(self, dataset_dir):
        """
        csv 파일을 경로에 맡게 불러 옵니다.
        """
        pd_dataset = pd.read_csv(dataset_dir)
        pd_dataset = remove_duplicate(pd_dataset)
        pd_dataset = use_sotype_token(pd_dataset)
        dataset = self.preprocessing_dataset(pd_dataset)
        
        return dataset

    def preprocessing_dataset(self, dataset):
        """
        처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
        """
        subject_entity = []
        object_entity = []
        for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset

    def tokenized_dataset(self, dataset, tokenizer):
        """
        tokenizer에 따라 sentence를 tokenizing 합니다.
        """
        concat_entity = []
        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
            temp = ''
            temp = e01 + '[SEP]' + e02
            concat_entity.append(temp)
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            )
        return tokenized_sentences

    def setup(self, stage='fit'):
        '''
        모델 사용 목적에 따른 데이터셋 생성
        stage: 모델 사용 목적(fit or test or predict)
        '''
        if stage == 'fit':
            train_dataset = self.load_data("./data/train.csv")
            #valid_datset = self.load_data("./data/valid.csv")
            train_label = label_to_num(train_dataset['label'].values)
            #valid_label = label_to_num(val_dataset['label].values)
            tokenized_train = self.tokenized_dataset(train_dataset, self.tokenizer)
            #tokenized_valid = self.tokenized_dataset(valid_dataset, self.tokenizer)

            self.train_dataset = Dataset(tokenized_train, train_label)
            #self.valid_dataset = Dataset(tokenized_valid, valid_label)
        elif stage == 'test':
            test_dataset = self.load_data("./data/train.csv")
            test_label = label_to_num(test_dataset['label'].values)
            tokenized_test = self.tokenized_dataset(test_dataset, self.tokenizer)

            self.test_dataset = Dataset(tokenized_test, test_label)
        elif stage == 'predict':
            predict_dataset = self.load_data("./data/test_data.csv")
            predict_label = list(map(int, predict_dataset["label"].values))
            tokenized_predict = self.tokenized_dataset(predict_dataset, self.tokenizer)
             
            self.predict_dataset = Dataset(tokenized_predict, predict_label)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers = self.num_workers)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    #     return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size)
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.num_workers)