from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import pandas as pd
import sys

import os
from tqdm import tqdm
import utils.DataPreprocessing as DataPreprocessing 
from collections import defaultdict

sys.path.append('/opt/ml/utils/LMKor/examples')

embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')


def data_augmentation(cfg):
    '''
    증강된 데이터를 기존 데이터에 합침
    '''
    original_data = pd.read_csv('./data/train.csv')
    methods = cfg['data_processing']
    new_data, method_names = get_new_data(original_data, methods)
    final_data = pd.concat([original_data, new_data], ignore_index=True)
    final_data['id'] = final_data.index
    os.makedirs('data/AugData', exist_ok=True)
    file_name =  '&'.join(method_names)+'_train.csv'
    final_data.to_csv(f'data/AugData/{file_name}')


def get_new_data(data, methods):
    '''
    증강 함수를 호출하고 증강 데이터를 하나로 합침
    '''
    data_bag = []
    method_names = []
    for method in methods.keys():
        if methods[method]:
            function = globals()[method]
            df = function(data)
            data_bag.append(df)
            method_names.append(method)

    new_data = pd.concat(data_bag, ignore_index=True)
    return new_data, method_names


def change_masked_word(predict_words, masked_sentence, sentence):
    new_sentences=[]
    for new_word in predict_words:
        new_sentence=masked_sentence.replace('<mask>', new_word)
        #유사도검사 후 일정 수치를 넘으면 new_sentences에 추가(현재 0.8)
        if get_similarity(new_sentence, sentence)>=0.8:
            new_sentences.append(new_sentence)
    return new_sentences


def data_augmentation(cfg):
    '''
    make new sentence and concat to vanilla data
    '''
    dataset = pd.read_csv('./data/train.csv')
    method = cfg['data_processing']['method']

    for data in dataset.itertuples():
        #'no_relation' data account half of data, no more 'no_relation' data
        if data.label != 'no_relation':
            new_data = get_new_data(data, method)
            dataset = pd.concat([dataset, new_data])
    dataset.to_csv(f'./eda/{method}.csv')


def delete_word(words, p):
    '''
    delete random words by prob 'p' but without entity tokens
    '''
    new_words = []
    for word in words:
        if '<subject_entity>' in word:
            new_words.append(word)
        elif '<object_entity>' in word:
            new_words.append(word)
        else:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
    return ' '.join(new_words)


def get_masked_sentence(tokens):
    '''
    mask one word in sentence but entity tokens can't be masked
    '''
    while True:
        replace_idx = random.randint(0, len(tokens)-1)
        if '<subject_entity>' not in tokens[replace_idx] and '<object_entity>' not in tokens[replace_idx]:
            tokens[replace_idx]='<mask>'
            break
    return ' '.join(tokens)


def get_new_data(data, method):
    '''
    make new data as dataframe
    '''
    entity = [eval(data.subject_entity)['word'], eval(data.object_entity)['word']]
    #change augmentation method by changing 'new_sentence'
    augmentation_method = {'random_deletion': random_deletion(data.sentence, entity),
                           'mlm_predict': mlm_augmentation(data.sentence, entity)}
    new_sentences = augmentation_method[method]
    new_data = pd.DataFrame({'id': [0]*len(new_sentences), 'sentence': new_sentences, 'subject_entity': [data.subject_entity]*len(new_sentences), 
                             'object_entity': [data.object_entity]*len(new_sentences), 'label': [data.label]*len(new_sentences), 'source': [data.source]*len(new_sentences)})
    return new_data


def get_similarity(new_sentence, sentence):
    '''
    get similarity of original sentence and new_sentence
    if sentences are same return -1
    '''
    if sentence==new_sentence:
        return 0
    new_sen_emb, sen_emb=embed_model.encode([new_sentence, sentence])
    cos_sim = np.dot(new_sen_emb, sen_emb)/(norm(new_sen_emb)*norm(sen_emb))
    return cos_sim


def mlm_augmentation(sentence, entity):
    '''
    make new sentence by using mlm model
    replace one word in sentence to prediction token of mlm model
    '''
    replaced_sentence = replace_entity_words_to_entity_token(sentence, entity)
    tokens = replaced_sentence.split()

    if len(tokens) < 5:
        return []
    masked_sentence = get_masked_sentence(tokens)
    masked_sentence = replace_entity_token_to_entity_words(masked_sentence, entity)
    predict_words = predict(masked_sentence)
    new_sentences = change_masked_word(predict_words, masked_sentence, sentence)
    return new_sentences


def random_deletion(sentence, entity, p_rd=0.1):
    '''
    make new sentence by random deletion method
    ※every sentence has at least 1 deletion
    ''' 
    sentence=replace_entity_words_to_entity_token(sentence, entity)
    words = [word for word in sentence.split(' ') if word != ""]
    new_sentences = []

    while True:
        new_sentence = delete_word(words, p_rd)
        new_sentence = replace_entity_token_to_entity_words(new_sentence, entity)
        if new_sentence != sentence:
            new_sentences.append(new_sentence)
            break
    return new_sentences


def replace_entity_words_to_entity_token(sentence: str, entity):
    '''
    replace entity words in sentence to special token
    '''
    sentence=sentence.replace(entity[0],'<subject_entity>')
    sentence=sentence.replace(entity[1],'<object_entity>')
    return sentence


def replace_entity_token_to_entity_words(sentence, entity):
    '''
    replace special token to entity words
    '''
    sentence=sentence.replace('<subject_entity>', entity[0])
    sentence=sentence.replace('<object_entity>', entity[1])
    return sentence


def random_insertion(original_df):
    '''
    sentence에 단어 삽입 (no_relation 라벨 제외)
    '''
    new_df = original_df[original_df['label']!='no_relation'].reset_index(drop=True)
    words_bag = []
    sentences = original_df[original_df['label']=='no_relation']['sentence'].tolist()
    for sentence in sentences:
        words_bag += sentence.split()
        
    new_sentences = []
    # for sentence in tqdm(new_df['sentence'].tolist(), desc='DataAug_Random_Insertion', mininterval=0.1):
    for sentence in tqdm(new_df['sentence'].tolist(), desc='DataAug_Random_Insertion'):
        length = len(sentence.split(' '))
        insert_id = random.randrange(length)
        new_sen = sentence.split(' ')
        new_sen.insert(insert_id, random.choice(words_bag))
        new_sentences.append(' '.join(new_sen))

    new_df['sentence'] = new_sentences
    return new_df


def random_swap(original_df):
    '''
    sentence에 임의 단어 자리 바꾸기(no_relation 라벨 제외)
    '''
    processed_df = DataPreprocessing.str_to_dict(original_df)
    new_df = processed_df[processed_df['label']!='no_relation'].reset_index(drop=True)

    sentences = new_df['sentence'].tolist()
    new_sentences = []
    for i in tqdm(range(len(sentences)), desc='DataAug_Random_Swap'):
        except1 = new_df['subject_entity'][i]['word']
        except2 = new_df['object_entity'][i]['word']
        new_df.loc[i,'subject_entity'] = str(new_df.loc[i, 'subject_entity'])
        new_df.loc[i, 'object_entity'] = str(new_df.loc[i, 'object_entity'])


        sen = sentences[i].replace(except1, '[sub]').replace(except2, '[ob]').split(' ')
        length= list(range(len(sen)))
        choice_lis = random.sample(length, 2)
        id1 = choice_lis[0]
        id2 = choice_lis[1]
        sen[id1], sen[id2] = sen[id2], sen[id1]
        new_sen = ' '.join(sen).replace( '[sub]', except1).replace( '[ob]', except2)
        new_sentences.append(new_sen)
        
    new_df['sentence'] = new_sentences

    return new_df




def sub_ob_swap(original_df):
    '''
    subject_entity와 object_entity 칼럼을 서로 바꿈
    '''
    new_df = original_df[original_df['label']!='no_relation']
    new_df['subject_entity'], new_df['object_entity'] = new_df['object_entity'], new_df['subject_entity']
    return new_df




