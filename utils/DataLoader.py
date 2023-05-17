import pandas as pd
import pickle as pickle
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer
from utils.Dataset import Dataset
from utils.Utils import label_to_num
from utils.DataPreprocessing import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils.DataPreprocessing import *

TYPE = {"ORG": "단체", "PER": "사람", "DAT": "날짜", "LOC": "위치", "POH": "기타", "NOH": "수량"}
LABEL = {'per:product':'<제작품>', 'org:number_of_employees/members':'명',
       'per:place_of_residence':'거주지', 'per:schools_attended':'학교',
       'per:place_of_birth':'출생지', 'org:founded_by':'창립'}

class DataLoader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, max_len, multi_sen, shuffle=True):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = 8
        self.max_length = max_len
        self.multi_sen = multi_sen

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name, additional_special_tokens=['#', '@']) #use_punct_mark 사용할땐 special token을 추가해주면됩니다.
    def load_data(self, dataset_dir):
        """
        csv 파일을 경로에 맡게 불러 옵니다.
        """
        pd_dataset = pd.read_csv(dataset_dir)
        pd_dataset = remove_duplicate(pd_dataset) #중복 제거
        pd_dataset = use_type_token(pd_dataset) #type token 추가
        #pd_dataset = use_punct_mark(pd_dataset)

        dataset = self.preprocessing_dataset(pd_dataset)
        return dataset

    def preprocessing_dataset(self, dataset):
        """
        처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
        """
        subject_entity = []
        object_entity = []
        if self.multi_sen:
            subject_type = []
            object_type = []
            for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
                sub = eval(i)['word']
                sub_type = eval(i)['type']
                obj = eval(j)['word']
                obj_type = eval(j)['type']

                subject_entity.append(sub)
                subject_type.append(sub_type)
                object_entity.append(obj)
                object_type.append(obj_type)
            
            out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],
                                        'subject_entity':subject_entity,'subject_type': subject_type,
                                        'object_entity':object_entity, 'object_type':object_type,
                                        'label':dataset['label']})
        
        else:
            for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
                i = eval(i)['word']
                j = eval(j)['word']

                subject_entity.append(i)
                object_entity.append(j)
            out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset

    def tokenized_dataset(self, dataset, tokenizer):
        """
        tokenizer에 따라 sentence를 tokenizing 합니다.
        """
        concat_entity = []
        if self.multi_sen:
            for e01, e02, e03, e04 in zip(
                dataset['subject_entity'],
                dataset['object_entity'],
                dataset['object_type'],
                dataset['label']
            ):
                temp = ''
                if e04 in LABEL.keys():
                    temp = f'이 문장에서 [{e02}]은 [{e01}]의 [{TYPE[e03]}][{LABEL[e04]}]이다.[SEP]'
                else:
                    temp = f'이 문장에서 [{e02}]은 [{e01}]의 [{TYPE[e03]}]이다.[SEP]'
                concat_entity.append(temp)

        else:
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
            max_length=self.max_length,
            add_special_tokens=True,
            )
        return tokenized_sentences
    
    def stratify(self):
        train_dataset = self.load_data("./data/train.csv")

        #train_size:valid_size = 8:2
        train_set, valid_set = train_test_split(train_dataset, test_size=0.2, random_state=42, shuffle=True, stratify=train_dataset['label'])
        
        train_label = label_to_num(train_set['label'].values)
        valid_label = label_to_num(valid_set['label'].values)

        tokenized_train = self.tokenized_dataset(train_set, self.tokenizer)
        tokenized_valid = self.tokenized_dataset(valid_set, self.tokenizer)
        
        self.train_dataset = Dataset(tokenized_train, train_label)
        self.valid_dataset = Dataset(tokenized_valid, valid_label) 


    def split(self):
        """
        Split the dataset into training and validation data.
        """
        train_dataset = self.load_data("./data/train.csv")
        """
        use pos tagging augmentation data.
        """
        pos_tag_dataset = pd.read_csv("./data/pos_tag2.csv", index_col = 0)
        pos_tag_dataset = use_type_token(pos_tag_dataset)
        pos_dataset = self.preprocessing_dataset(pos_tag_dataset)

        #train_size:valid_size = 8:2
        train_set, valid_set = train_test_split(train_dataset, test_size=0.2, random_state=42, shuffle=True)
        
        train_label = label_to_num(train_set['label'].values)
        valid_label = label_to_num(valid_set['label'].values)

        tokenized_train = self.tokenized_dataset(train_set, self.tokenizer)
        tokenized_valid = self.tokenized_dataset(valid_set, self.tokenizer)
        
        self.train_dataset = Dataset(tokenized_train, train_label)
        self.valid_dataset = Dataset(tokenized_valid, valid_label) 

    def nonSplit(self):
        """
        without splitting
        """
        train_dataset = self.load_data("./data/train.csv")
        #valid_datset = self.load_data("./data/valid.csv")

        train_label = label_to_num(train_dataset['label'].values)
        #valid_label = label_to_num(val_dataset['label'].values)

        tokenized_train = self.tokenized_dataset(train_dataset, self.tokenizer)
        #tokenized_valid = self.tokenized_dataset(valid_dataset, self.tokenizer)

        self.train_dataset = Dataset(tokenized_train, train_label)
        #self.valid_dataset = Dataset(tokenized_valid, valid_label)

    def setup(self, stage='fit'):
        '''
        모델 사용 목적에 따른 데이터셋 생성
        stage: 모델 사용 목적(fit or test or predict)
        '''
        if stage == 'fit':
            self.stratify()
            #self.split()
            #self.nonSplit()
        
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
        #return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
        return torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers = self.num_workers)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = self.num_workers)