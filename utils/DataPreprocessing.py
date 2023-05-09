def remove_duplicate(dataset):
    duplicate = dataset[dataset.duplicated(subset = ['sentence', 'subject_entity', 'object_entity'], keep = False)]
    
    id = []
    for name, group in duplicate.groupby(['sentence', 'subject_entity', 'object_entity']):
        remov = group['label'].duplicated(keep = 'last')
        for key, value in remov.items():
            if value == False:
                id.append(key)
                
    duplicate.drop(id, axis = 0, inplace = True)
    
    return duplicate

def use_token(dataset):
    start_token = '[ENT]'
    end_token = '[/ENT]'
    
    for i in range(len(dataset)):
        subject_ent = eval(dataset['subject_entity'][i])
        object_ent = eval(dataset['object_entity'][i])
        sen = dataset['sentence'][i]
        if subject_ent['start_idx'] < object_ent['start_idx']:
            dataset['sentence'][i] = sen[:subject_ent['start_idx']] + start_token + subject_ent['word'] + end_token + sen[subject_ent['end_idx']+1:]
            sen = dataset['sentence'][i]
            dataset['sentence'][i] = sen[:object_ent['start_idx']+11] + start_token + object_ent['word'] + end_token + sen[object_ent['end_idx']+12:]
        elif subject_ent['start_idx'] > object_ent['start_idx']:
            dataset['sentence'][i] = sen[:object_ent['start_idx']] + start_token + object_ent['word'] + end_token + sen[object_ent['end_idx']+1:]
            sen = dataset['sentence'][i]
            dataset['sentence'][i] = sen[:subject_ent['start_idx']+11] + start_token + subject_ent['word'] + end_token + sen[subject_ent['end_idx']+12:]
        
    return dataset