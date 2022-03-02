from dataloader import DataGenerator
import torch
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import torchvision.transforms as transforms
import torchvision.datasets as dset
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import regularizers
import cv2
import matplotlib.pyplot as plt



def read_from_list(list_of_files, data_root):
       
        def read_text_file(fname, data_root=''):
            print('I,m here',fname)
            f = open(fname, 'r')
            lines = f.readlines()
            file_address = []
            quality_label = []
            view_label = []
            quality_label_t2 = []

            for line in lines:
                parts = line.split('\n')
                line = parts[0][1:]
                parts = line.split(',')
                fa = data_root + parts[0]
                if 'single' == 'inter' and int(parts[3]) == -1:
                    continue
                if int(parts[2])<=13:
                    file_address.append(fa)
                    quality_label.append(int(parts[1]))
                    view_label.append(int(parts[2]))
                    if len(parts) == 4:
                        quality_label_t2.append(int(parts[3]))
                    else:
                        quality_label_t2.append(int(-1))

            return file_address, quality_label, view_label, quality_label_t2

        file_address = []
        quality_label = []
        view_label = []
        quality_label_t2 = []

        for fname in list_of_files:
            fa, ql, vl, ql2 = read_text_file(fname, data_root)
            file_address += fa
            quality_label += ql
            view_label += vl
            quality_label_t2 += ql2

        quality_label = np.asarray(quality_label)
        view_label = np.asarray(view_label)
        quality_label_t2 = np.asarray(quality_label_t2)
        return {'file_address': file_address, 'quality_label': quality_label, 'view_label': view_label,
        'quality_label_t2': quality_label_t2, 'num_files': len(file_address)}
#
