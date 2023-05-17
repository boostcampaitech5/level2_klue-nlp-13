from ast import literal_eval

def remove_duplicate(dataset):
    '''
    Delete duplicate data with different label
    '''
    duplicate = dataset[dataset.duplicated(subset = ['sentence', 'subject_entity', 'object_entity'], keep = False)]
    
    id = []
    for name, group in duplicate.groupby(['sentence', 'subject_entity', 'object_entity']):
        remov = group['label'].duplicated(keep = 'last')
        for key, value in remov.items():
            if value == False:
                id.append(key)
                
    dataset.drop(id, axis = 0, inplace = True)
    
    return dataset

def str_to_dict(dataset):
    '''
    Entity data, which is a string type, replace to dictionary type
    '''
    def func(obj):
        List = literal_eval(obj)
        return List
    
    out = dataset.copy()
    out['subject_entity'] = dataset['subject_entity'].apply(func)
    out['object_entity'] = dataset['object_entity'].apply(func)
    
    return out
    
def use_ent_token(dataset):
    '''
    Use [ENT] token to indicate entity before and after entity word
    '''
    start_token = '[ENT]'
    end_token = '[/ENT]'

    out_dataset = str_to_dict(dataset)
    sens =[]
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]
        object_ent = out_dataset['object_entity'][i]
        sen = out_dataset['sentence'][i]
        if subject_ent['start_idx'] < object_ent['start_idx']:
            sen = sen[:subject_ent['start_idx']] + start_token + subject_ent['word'] + end_token + sen[subject_ent['end_idx']+1:]
            sen = sen[:object_ent['start_idx']+11] + start_token + object_ent['word'] + end_token + sen[object_ent['end_idx']+12:]
        elif subject_ent['start_idx'] > object_ent['start_idx']:
            sen = sen[:object_ent['start_idx']] + start_token + object_ent['word'] + end_token + sen[object_ent['end_idx']+1:]
            sen = sen[:subject_ent['start_idx']+11] + start_token + subject_ent['word'] + end_token + sen[subject_ent['end_idx']+12:]
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset

def use_type_token(dataset):
    '''
    Use [type] token to indicate entity before and after entity word
    '''
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f'[{sub_type}]{subject_ent}[/{sub_type}]')
        sen = sen.replace(object_ent, f'[{obj_type}]{object_ent}[/{obj_type}]')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset

def use_sotype_token(dataset):
    '''
    Use type tokens that distinguish between subject and object before and after entity words
    '''
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]
        object_ent = out_dataset['object_entity'][i]
        sub_type = subject_ent['type']
        obj_type = subject_ent['type']
        sen = out_dataset['sentence'][i]
        if subject_ent['start_idx'] < object_ent['start_idx']:
            sen = sen[:subject_ent['start_idx']] + f'[S-{sub_type}]' + subject_ent['word'] + f'[/S-{sub_type}]' + sen[subject_ent['end_idx']+1:]
            sen = sen[:object_ent['start_idx']+15] + f'[O-{obj_type}]' + object_ent['word'] + f'[/O-{obj_type}]' + sen[object_ent['end_idx']+16:]
        elif subject_ent['start_idx'] > object_ent['start_idx']:
            sen = sen[:object_ent['start_idx']] + f'[O-{obj_type}]' + object_ent['word'] + f'[/O-{obj_type}]' + sen[object_ent['end_idx']+1:]
            sen = sen[:subject_ent['start_idx']+15] + f'[S-{sub_type}]' + subject_ent['word'] + f'[/S-{sub_type}]' + sen[subject_ent['end_idx']+16:]
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset

def use_typed_entity_mark(dataset):
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]
        object_ent = out_dataset['object_entity'][i]
        sub_type = subject_ent['type']
        obj_type = subject_ent['type']
        sen = out_dataset['sentence'][i]
        if subject_ent['start_idx'] < object_ent['start_idx']:
            sen = sen[:subject_ent['start_idx']] + f'<S:{sub_type}> ' + subject_ent['word'] + f' </S:{sub_type}>' + sen[subject_ent['end_idx']+1:]
            sen = sen[:object_ent['start_idx']+11] + f'<O:{obj_type}> ' + object_ent['word'] + f' </O:{obj_type}>' + sen[object_ent['end_idx']+12:]
        elif subject_ent['start_idx'] > object_ent['start_idx']:
            sen = sen[:object_ent['start_idx']] + f'<O:{obj_type}> ' + object_ent['word'] + f' </O:{obj_type}>' + sen[object_ent['end_idx']+1:]
            sen = sen[:subject_ent['start_idx']+11] + f'<S:{sub_type}> ' + subject_ent['word'] + f' </S:{sub_type}>' + sen[subject_ent['end_idx']+12:]
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset

def use_punct_mark(dataset):
    """
    Mark entity types with punctuations
    """
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]
        object_ent = out_dataset['object_entity'][i]
        sub_type = subject_ent['type']
        obj_type = subject_ent['type']
        sen = out_dataset['sentence'][i]
        if subject_ent['start_idx'] < object_ent['start_idx']:
            sen = sen[:subject_ent['start_idx']] +' @ * ' +f'[{sub_type}]'+' * ' + subject_ent['word'] + ' @ ' + sen[subject_ent['end_idx']+1:]
            sen = sen[:object_ent['start_idx']+14] + ' # ^ ' +f'[{obj_type}]' +' ^ '+ object_ent['word'] + ' # ' + sen[object_ent['end_idx']+15:]
        elif subject_ent['start_idx'] > object_ent['start_idx']:
            sen = sen[:object_ent['start_idx']] + ' # ^ ' +f'[{obj_type}]' +' ^ '+ object_ent['word'] + ' # ' + sen[object_ent['end_idx']+1:]
            sen = sen[:subject_ent['start_idx']+14] +' @ * ' +f'[{sub_type}]'+' * ' + subject_ent['word'] + ' @ ' + sen[subject_ent['end_idx']+15:]
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset
    