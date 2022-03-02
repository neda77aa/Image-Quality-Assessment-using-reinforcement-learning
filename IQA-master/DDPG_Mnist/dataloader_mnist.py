import numpy as np
from tensorflow import  keras
import tensorflow.keras
import tensorflow

class DataGenerator(tensorflow.keras.utils.Sequence):
    
    
    def __init__(self, data, labels, batch_size=64, dim=(36,36), n_channels=1,
                 n_classes=2, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
              

            X= self.data[index*self.batch_size:(index+1)*self.batch_size]
            y= self.labels[index*self.batch_size:(index+1)*self.batch_size]

            return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)

