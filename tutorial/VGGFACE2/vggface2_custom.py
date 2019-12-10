
import os
from glob import glob

from PIL import Image
import numpy as np


def vgg_load(select_classes=150, start_ind=0):
    tr_data_path = 'd:/dataset/vggface2_custom/trainset'
    va_data_path = 'd:/dataset/vggface2_custom/validset'

    start_label = start_ind
    end_label = start_label + select_classes

    label_list = sorted(os.listdir(tr_data_path))

    selected_label_list = label_list[start_label:end_label]    

    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    
    label_vocab = dict()
    for i in range(len(selected_label_list)):
        label_vocab[selected_label_list[i]] = i


    for label in selected_label_list:
        tr_file_list = os.listdir(os.path.join(tr_data_path, label))
        va_file_list = os.listdir(os.path.join(va_data_path, label))

        #for i in range(len(tr_file_list)):
        for i in range(450):
            file_path = os.path.join(tr_data_path, label, tr_file_list[i])
            image = np.array(Image.open(file_path))
            train_x.append(image)
            train_y.append(label_vocab[label])

        
        for i in range(len(va_file_list)):
            file_path = os.path.join(va_data_path, label, va_file_list[i])
            image = np.array(Image.open(file_path))
            valid_x.append(image)
            valid_y.append(label_vocab[label])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)

    return (train_x, train_y), (valid_x, valid_y)


if __name__ == '__main__':
    (train_x, train_y), (valid_x, valid_y) = vgg_load()
    print('train_x shape : ', train_x.shape)
    print('train_y shape : ', train_y.shape)
    print('valid_x shape : ', valid_x.shape)
    print('valid_y shape : ', valid_y.shape)


    



