import numpy as np
from tensorflow.keras.models import Sequential
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
from tensorflow.keras import layers
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.models import load_model
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    
    def read_from_list(list_of_files, data_root):
       
        def read_text_file(fname, data_root=''):
            print(fname)
            f = open(fname, 'r')
            lines = f.readlines()
            file_address = []
            quality_label = []
            view_label = []
            quality_label_t2 = []

            for line in lines:
                parts = line.split('\n')
                line = parts[0]
                parts = line.split(',')
                fa = data_root + parts[0]
                if not os.path.isfile(fa):
                    print(fa + ' does not exist')
                    continue
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
        
    def __init__(self,data_dict, batch_size=32, dim=(224,224), n_channels=1,
                 n_classes=14, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_dict = data_dict
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.len = self.__len__() 
        self.on_epoch_end()
        np.random.seed(1)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data_dict['num_files']/self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        filename_temp = [self.data_dict['file_address'][k] for k in indexes]
        view_temp = [self.data_dict['view_label'][k] for k in indexes]
       

        X , y  = self.__data_generation(filename_temp , view_temp) 
        return X , y 
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.data_dict['num_files'])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
    def __data_generation(self, filename_temp , view_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        for i, path in enumerate(filename_temp):
            try:
                matfile = sio.loadmat(path, verify_compressed_data_integrity=False)
            
            except TypeError:
                raise TypeError()

            d = matfile['Patient']['DicomImage'][0][0]

            r = np.random.randint(0, d.shape[2])
            d = d[:, :, r]
            image=Image.fromarray(d)
            image=image.resize((224, 224))
            X[i,:,:,:] = np.array(image).reshape((224, 224,1))

        return X, tensorflow.keras.utils.to_categorical(view_temp, num_classes=self.n_classes)
    

    
    