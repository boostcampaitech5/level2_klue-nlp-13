import random
import pandas as pd


def delete_word(words, entity, p):
    '''
    delete random words by prob 'p' but without entity words
    '''
    new_words = []
    for word in words:
        if '<subject_entity>' in word:
            new_words.append(word.replace('<subject_entity>',entity[0]))
        elif '<object_entity>' in word:
            new_words.append(word.replace('<object_entity>', entity[1]))
        else:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
    return new_words


def get_new_data(data, method):
    '''
    make new data as dataframe
    '''
    entity = [data.subject_entity.split(',')[0].split(':')[1][2:-1], data.object_entity.split(',')[0].split(':')[1][2:-1]]
    
    #change augmentation method by changing 'new_sentence'
    augmentation_method = {'random_deletion': random_deletion(data.sentence, entity)}
    new_sentences = augmentation_method[method]
    new_data = pd.DataFrame({'id': [0], 'sentence':new_sentences, 'subject_entity': [data.subject_entity], 
                             'object_entity': [data.object_entity], 'label': [data.label], 'source': [data.source]})
    return new_data


def random_deletion(sentence, entity, p_rd=0.1):
    '''
    make new sentence by random deletion method
    ※every sentence has at least 1 deletion
    ''' 
    sentence=sentence.replace(entity[0],'<subject_entity>')
    sentence=sentence.replace(entity[1],'<object_entity>')

    words = sentence.split(' ')
    words = [word for word in words if word != ""]

    while True:
        new_words = delete_word(words, entity, p_rd)
        new_sentence = ' '.join(new_words)

        if new_sentence != sentence:
            break
    return new_sentence

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