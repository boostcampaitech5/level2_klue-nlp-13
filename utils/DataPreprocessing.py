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