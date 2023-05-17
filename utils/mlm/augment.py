import copy
import torch
import tqdm
import argparse
import pandas as pd
import random
from typing import Union
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM

import sys
sys.path.append('./utils/mlm')
from util import mask_tokens




def load_tuned_model():
    dev = torch.device('cuda:0')
    model = AutoModelForMaskedLM.from_pretrained("{model_path}")
    tokenizer = AutoTokenizer.from_pretrained("{model_path}")
    model.resize_token_embeddings(tokenizer.vocab_size+2)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model.to(dev)
    return model, tokenizer, dev

def tokenize(tokenizer:AutoTokenizer, sent:str):
    encoded_dict = tokenizer(
        sent,
        add_special_tokens = True,
        return_attention_mask = True,
        return_tensors = "pt"
    )
    input_id, attention_mask = encoded_dict.input_ids, encoded_dict.attention_mask

    return input_id, attention_mask

def is_same_token_type(org_token:str, candidate:str) -> bool:
    '''
    후보 필터링 조건을 만족하는지 확인
    - 후보와 원 토큰의 타입을 문장부호와 일반 토큰으로 나누어 같은 타입에 속하는지 확인
    '''
    res = False
    if org_token[0]=="#" and org_token[2:].isalpha()==candidate.isalpha():
        res = True
    elif candidate[0]=="#" and org_token.isalpha()==candidate[2:].isalpha():
        res = True
    elif candidate[0]=="#" and org_token[0]=="#" and org_token[2:].isalpha()==candidate[2:].isalpha():
        res = True
    elif org_token.isalpha()==candidate.isalpha() and (candidate[0]!="#" and org_token[0]!="#"):
        res = True

    return res

def candidate_filtering(tokenizer:AutoTokenizer,
                        input_ids:list,
                        idx:int,
                        org:int,
                        candidates:Union[list, torch.Tensor]) -> int:
    '''
    후보 필터링 조건에 만족하는 최적의 후보 선택
    1. 원래 토큰과 후보 토큰이 같은 타입(is_same_token_type 참고)
    2. 현 위치 앞 혹은 뒤에 동일한 토큰이 있지 않음
    '''

    org_token = tokenizer.convert_ids_to_tokens([org])[0]
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidates.cpu().tolist())

    for rank, token in enumerate(candidate_tokens):
        if org_token!=token and is_same_token_type(org_token, token):
            if input_ids[idx-1]==candidates[rank] or input_ids[idx+1]==candidate_tokens[rank]:
                continue
            return candidates[rank]

    return org

def augment_one_sent(model:AutoModelForMaskedLM,
                    tokenizer:AutoTokenizer,
                    sent:str,
                    dev:Union[str, torch.device]) -> str:
    '''
    한 문장에 랜덤으로 마스킹을 적용하여 새로운 문장을 생성(증강)

    args:
        model(AutoModelForMaskedLM)     : finetuned model
        tokenizer(AutoTokenizer)
        sent(str)                       : 증강할 문장
        dev(str or torch.device)
            - k(int, default=5) : 사용할 후보의 개수. k개의 후보 적절한 토큰이 없을 경우 원래 토큰 그대로 유지
            - threshold(float, default=0.95) : 확률 필터링에 사용할 임계치.
                                               마스크에 대해서 특정 후보 토큰을 생성할 확률이 임계치보다 클 경우에는 별도의 필터링 없이 후보를 그대로 사용.
           -  mlm_prob(float, default=0.15) : 마스킹 비율
        
    return:
        (str) : 증강 문장
    '''
    k = 3
    threshold = 0.95
    mlm_prob = 0.15

    model.eval()

    input_id, attention_mask  = tokenize(tokenizer, sent)
    org_ids = copy.deepcopy(input_id[0])

    masked_input_id, _ = mask_tokens(tokenizer, input_id, mlm_prob, do_rep_random=False)
    while masked_input_id.cpu().tolist()[0].count(tokenizer.mask_token_id) < 1:
        masked_input_id, _ = mask_tokens(tokenizer, input_id, mlm_prob, do_rep_random=False)
    
    with torch.no_grad():
        masked_input_id, attention_mask = masked_input_id.to(dev), attention_mask.to(dev)
        output = model(masked_input_id, attention_mask = attention_mask)
        logits = output["logits"][0]

    copied = copy.deepcopy(masked_input_id.cpu().tolist()[0])
    for i in range(len(copied)):
        if copied[i] == tokenizer.mask_token_id:
            org_token = org_ids[i]
            prob = logits[i].softmax(dim=0)
            probability, candidates = prob.topk(k)
            if probability[0]<threshold:
                res = candidate_filtering(tokenizer, copied, i, org_token, candidates)
            else:
                res = candidates[0]
            copied[i] = res

    copied = tokenizer.decode(copied, skip_special_tokens=True)

    return copied

if __name__ == "__main__":
    
    model, tokenizer, dev = load_tuned_model() 

    datas = pd.read_csv('./data/unofficial_train.csv')

    idx = datas.iloc[-1]['id']+1

    for data in datas.itertuples():
        if data.label != 'no_relation':
            input_sen=data.sentence
            subject_entity = eval(data.subject_entity)
            object_entity = eval(data.object_entity)
            entity = [subject_entity['word'], object_entity['word']]
            input_sen=input_sen.replace(entity[0], '<subject_entity>')
            input_sen=input_sen.replace(entity[1], '<object_entity>')
            augmented = augment_one_sent(model, tokenizer, input_sen, dev)
            new_sentence = augmented.replace('<subject_entity>', entity[0]).replace('<object_entity>', entity[1])
            subject_entity['start_idx']=new_sentence.find(entity[0])
            subject_entity['end_idx']=subject_entity['start_idx']+len(entity[0])-1
            object_entity['start_idx']=new_sentence.find(entity[1])
            object_entity['end_idx']=object_entity['start_idx']+len(entity[1])-1


            temp = pd.DataFrame({'id': [str(idx)], 'sentence': [new_sentence], 'subject_entity': [subject_entity], 
                                'object_entity': [object_entity], 'label': [data.label], 'source': [data.source]})
            datas = pd.concat([datas, temp])
            idx += 1
            print(new_sentence)
    datas.to_csv('./a.csv',index=False)