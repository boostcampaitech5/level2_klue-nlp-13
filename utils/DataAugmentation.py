from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import pandas as pd
import sys

sys.path.append('./utils/mlm')
from augment import load_tuned_model, augment_one_sent

embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
mlm_model, mlm_tokenizer, mlm_dev = load_tuned_model()
idx = 0

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
    global idx
    dataset = pd.read_csv('./data/unofficial_train.csv')
    method = cfg['data_processing']['method']

    idx = dataset.iloc[-1]['id']+1

    for data in dataset.itertuples():
        #'no_relation' data account half of data, no more 'no_relation' data
        # if data.label != 'no_relation':
        #     new_data = get_new_data(data, method)
        #     dataset = pd.concat([dataset, new_data])
        new_data = get_new_data(data, method)
        dataset = pd.concat([dataset, new_data])
    dataset.to_csv(f'./eda/{method}.csv', index=False)


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

def get_new_data(data, method):
    '''
    make new data as dataframe
    '''
    global idx
    subject_entity = eval(data.subject_entity)
    object_entity = eval(data.object_entity)
    entity = [subject_entity['word'], object_entity['word']]
        
    #change augmentation method by changing 'new_sentence'
    augmentation_method = {'random_deletion': random_deletion(data.sentence, entity),
                           'mlm_predict': mlm_augmentation(data.sentence, entity, mlm_model, mlm_tokenizer, mlm_dev)}
    new_sentences = augmentation_method[method]
    new_data=pd.DataFrame()
    for new_sentence in new_sentences:
        subject_entity['start_idx']=new_sentence.find(entity[0])
        subject_entity['end_idx']=subject_entity['start_idx']+len(entity[0])-1
        object_entity['start_idx']=new_sentence.find(entity[1])
        object_entity['end_idx']=object_entity['start_idx']+len(entity[1])-1
        temp = pd.DataFrame({'id': [str(idx)], 'sentence': [new_sentence], 'subject_entity': [subject_entity], 
                             'object_entity': [object_entity], 'label': [data.label], 'source': [data.source]})
        new_data = pd.concat([new_data, temp])
        idx += 1
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