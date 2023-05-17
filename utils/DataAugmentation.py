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
#sys.path.append('./utils/mlm')
#from augment import load_tuned_model, augment_one_sent

embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
#mlm_model, mlm_tokenizer, mlm_dev = load_tuned_model()

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

def mlm_augmentation(sentence, entity, mlm_model, tokenizer, dev, rep=1):
    '''
    make new sentence by using mlm model
    replace one word in sentence to prediction token of mlm model
    '''
    new_sentences = []
    for r in range(rep):
        replaced_sentence = replace_entity_words_to_entity_token(sentence, entity)
        new_sentence = augment_one_sent(mlm_model, tokenizer, replaced_sentence, dev)
        new_sentence = replace_entity_token_to_entity_words(new_sentence, entity)
        if get_similarity(new_sentence, sentence)>0.9 and new_sentence not in new_sentences:
            new_sentences.append(new_sentence)
    return new_sentences


def random_deletion(sentence, entity, p_rd=0.1):
    '''
    make new sentence by random deletion method
    ※every sentence has at least 1 deletion
    ''' 
    random.seed(42)
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




def same_type_swap(original_df):
    '''
    type, Label이 같은 데이터끼리 Subject_entity와 Object_entity를 sentence에 넣어서 증강(no_relation 라벨 제외)
    '''
    processed_df = DataPreprocessing.str_to_dict(original_df)  
    df = processed_df[processed_df['label']!='no_relation']
    df.loc[:,'subject_entity_type']= df['subject_entity'].apply(lambda x: x['type'])
    df.loc[:,'object_entity_type'] = df['object_entity'].apply(lambda x: x['type'])
    df_gb = df.groupby(['label','subject_entity_type', 'object_entity_type'])

    dic = defaultdict(list)
    for group in tqdm(df_gb.groups.keys(), desc='DataAug_Same_type_swap'):
        group_df = df_gb.get_group(group) 
        if len(group_df) >1 and len(group_df) < 200:
            for i in group_df.index:
                original_sub = group_df['subject_entity'][i]
                original_ob = group_df['object_entity'][i]
                sen =  group_df['sentence'][i].replace(original_sub['word'], '[sub]').replace(original_ob['word'], '[ob]')
                for j in group_df.index:
                    if i!=j:
                        new_sub = group_df['subject_entity'][j]
                        new_ob = group_df['object_entity'][j]
                        replaced_sen = sen.replace('[sub]', new_sub['word']).replace('[ob]', new_ob['word'])
                        label = group_df['label'][j]
                        dic['sentence'].append(replaced_sen)
                        dic['subject_entity'].append(str(new_sub))
                        dic['object_entity'].append(str(new_ob))
                        dic['label'].append(label)
    dic['id'] = 0
    dic['source'] ='agumentation'
    new_df = pd.DataFrame(dic)
    return new_df





