import PIL
from PIL import Image
import glob
import numpy as np
from sewar.full_ref import mse


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


def get_average_train_val_class_split(train_data_object, k_fold_object):
    train_dictionary = {}
    val_dictionary = {}

    for train_indices, val_indices in k_fold_object.split(np.arange(len(train_data_object)),
                                                          np.array(train_data_object.targets, dtype='int32')):
        for train_id in train_indices:
            if train_data_object.targets[train_id] not in train_dictionary.keys():
                train_dictionary[train_data_object.targets[train_id]] = 1
            else:
                train_dictionary[train_data_object.targets[train_id]] += 1

        for val_id in val_indices:
            if train_data_object.targets[val_id] not in val_dictionary.keys():
                val_dictionary[train_data_object.targets[val_id]] = 1
            else:
                val_dictionary[train_data_object.targets[val_id]] += 1

    train_dictionary = {key: round(train_dictionary[key] / 5) for key in train_dictionary}
    val_dictionary = {key: round(val_dictionary[key] / 5) for key in val_dictionary}

    return train_dictionary, val_dictionary


def load_images(data_object, class_index=None):
    if class_index is not None:
        image_paths = []
        for path_class_tuple in data_object.imgs:
            if path_class_tuple[1] == class_index:
                image_paths.append(path_class_tuple[0])
    else:
        image_paths = [x[0] for x in data_object.imgs]
    data = []

    for image_path in image_paths:
        image = PIL.Image.open(image_path)
        data.append(image)

    return data


def save_average_image(images, file_name):
    images = np.array([np.array(im) for im in images])
    average_image = np.average(images, axis=0)
    result = Image.fromarray(average_image.astype('uint8'))
    result.save(file_name)


def save_all_average_class_images(data_object):
    for key, value in data_object.class_to_idx.items():
        class_name = key
        class_index = value
        train_class_images = load_images(data_object, class_index=class_index)
        save_average_image(train_class_images, class_name + '.jpg')


def calculate_mse_between_average_images(class_array):
    class_number = len(class_array)
    mse_matrix = np.zeros((class_number, class_number))
    average_images = []

    for class_name in class_array:
        file_name = class_name + '.jpg'
        img = PIL.Image.open(file_name)
        img = np.array(img)
        average_images.append(img)

    for i in range(class_number):
        for j in range(class_number):
            mse_matrix[i,j] = mse(average_images[i],average_images[j])

    return mse_matrix

