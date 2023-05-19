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
    out_dataset = str_to_dict(dataset)
    sens =[]
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f'[ENT]{subject_ent}[/ENT]')
        sen = sen.replace(object_ent, f'[ENT]{object_ent}[/ENT]')
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
       subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f'[S-{sub_type}]{subject_ent}[/{sub_type}]')
        sen = sen.replace(object_ent, f'[O-{obj_type}]{object_ent}[/{obj_type}]')
        sens.append(sen)
        
    return dataset

def use_typed_entity_mark(dataset):
    out_dataset = str_to_dict(dataset)
    sens = []
    for i in list(out_dataset['id']):
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f'<S-{sub_type}> {subject_ent} </S-{sub_type}>')
        sen = sen.replace(object_ent, f'<O-{obj_type}> {object_ent} </O-{obj_type}>')
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
        subject_ent = out_dataset['subject_entity'][i]['word']
        object_ent = out_dataset['object_entity'][i]['word']
        sub_type = out_dataset['subject_entity'][i]['type']
        obj_type = out_dataset['object_entity'][i]['type']
        sen = out_dataset['sentence'][i]
        sen = sen.replace(subject_ent, f' @ * {sub_type} * {subject_ent} @ ')
        sen = sen.replace(object_ent, f' # ^ {obj_type} ^ {object_ent} # {obj_type}')
        sens.append(sen)

    dataset['sentence'] = sens
        
    return dataset
    