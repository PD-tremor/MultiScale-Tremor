
"""
Multi-Scale Motion Representation Learning for Video-based Parkinson's Disease Tremor Assessment
"""
import os

ROOT_DATASET = './ssd/'  # '/data/jilin/'

def return_tremor(modality):

    filename_categories = './category.txt'


    if modality == 'Flow':

        ROOT_DATASET = './'
        root_data = os.path.join(
            ROOT_DATASET,
            './'
        )
        LABEL_PATH='./'
        filename_imglist_train = os.path.join(LABEL_PATH, 'train.txt')
        filename_imglist_val = os.path.join(LABEL_PATH, 'val.txt')
        prefix = '{:06d}.jpg'


    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix







def return_dataset(dataset, modality):
    dict_single = {'tremor': return_tremor }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(ROOT_DATASET, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:  # number of categories
        categories = [None] * file_categories
    n_class = len(categories)
    print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix


