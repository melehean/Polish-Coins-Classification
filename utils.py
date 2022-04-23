import PIL
from PIL import Image
import glob
import numpy as np


def count_resolution(path):
    files = glob.glob(path + '/**/*.jpg', recursive=True)
    dictionary = {}

    for file in files:
        img = PIL.Image.open(file)
        width, height = img.size
        if width > height:
            temp = width
            width = height
            height = temp
        if (width, height) not in dictionary.keys():
            dictionary[(width, height)] = 1
        else:
            dictionary[(width, height)] += 1

    return dictionary


def get_average_train_val_class_split(train_data, k_fold_object):
    train_dictionary = {}
    val_dictionary = {}

    for train_indices, val_indices in k_fold_object.split(np.arange(len(train_data)), np.array(train_data.targets, dtype='int32')):
        for train_id in train_indices:
            if train_data.targets[train_id] not in train_dictionary.keys():
                train_dictionary[train_data.targets[train_id]] = 1
            else:
                train_dictionary[train_data.targets[train_id]] += 1

        for val_id in val_indices:
            if train_data.targets[val_id] not in val_dictionary.keys():
                val_dictionary[train_data.targets[val_id]] = 1
            else:
                val_dictionary[train_data.targets[val_id]] += 1

    train_dictionary = {key: round(train_dictionary[key] / 5) for key in train_dictionary}
    val_dictionary = {key: round(val_dictionary[key] / 5) for key in val_dictionary}

    return train_dictionary, val_dictionary
