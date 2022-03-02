from Interface import PPOInterface
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
import numpy as np
import os
from dataloader_mnist import DataGenerator
import matplotlib.pyplot as plt
import cv2



img_shape = (36, 36, 1)



#Data Generator 
###############################################################

params_train = {'dim': (36,36),
          'batch_size': 64,
          'n_classes':  2,
          'n_channels': 1,
          'shuffle': False}

# Parameters_validation
params_val = {'dim': (36,36),
          'batch_size': 1,
          'n_classes':  2,
          'n_channels': 1,
          'shuffle': False}

params_holdout = {'dim': (36,36),
          'batch_size': 1,
          'n_classes':  2,
          'n_channels': 1,
          'shuffle': True}



numpy_zip = np.load(r'pneumoniamnist_corrupted.npz')

x_train0, y_train = numpy_zip['x_train'], numpy_zip['y_train']
x_val0, y_val = numpy_zip['x_val'], numpy_zip['y_val']
x_holdout0, y_holdout = numpy_zip['x_holdout'], numpy_zip['y_holdout']

num_train_samples = len(y_train)
num_val_samples = len(y_val)
num_holdout_samples = len(y_holdout)


x_train = np.empty((num_train_samples,img_shape[0],img_shape[1],img_shape[2]), float) 
x_val = np.empty((num_val_samples,img_shape[0],img_shape[1],img_shape[2]), float) 
x_holdout = np.empty((num_holdout_samples,img_shape[0],img_shape[1],img_shape[2]), float) 


for i in range(num_train_samples): x_train [i,:,:,:] = np.expand_dims(cv2.resize(x_train0[i,:,:,:], img_shape[0:2], interpolation = cv2.INTER_AREA),axis=-1)
for i in range(num_val_samples): x_val [i,:,:,:] = np.expand_dims(cv2.resize(x_val0[i,:,:,:], img_shape[0:2], interpolation = cv2.INTER_AREA),axis=-1)
for i in range(num_holdout_samples): x_holdout [i,:,:,:] = np.expand_dims(cv2.resize(x_holdout0[i,:,:,:], img_shape[0:2], interpolation = cv2.INTER_AREA),axis=-1)


del x_train0
del x_val0
del x_holdout0


training_generator = DataGenerator(x_train, y_train, **params_train)
validation_generator = DataGenerator(x_val, y_val, **params_val)
holdout_generator = DataGenerator(x_holdout, y_holdout, **params_holdout)
###############################################################





#Task Predictor 
###############################################################
def build_task_predictor(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(2, activation='sigmoid')(x)
    return keras.Model(inputs=inputs, outputs=outputs)



task_predictor = build_task_predictor(img_shape)
task_predictor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # speciffy the loss and metric used to train target net and controller respectively
###############################################################






# Controller PPO
###############################################################
interface = PPOInterface(training_generator, validation_generator, holdout_generator, task_predictor, img_shape,load_models=False)
num_train_episodes =500
interface.train(num_train_episodes)


save_dir = 'temp'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

controller_save_path = r'temp/train_session_PPO_Mnist_Generator_controller'
task_predictor_save_path = r'temp/train_session_PPO_Mnist_Generator_predictor'
interface.save(controller_save_path=controller_save_path,
               task_predictor_save_path=task_predictor_save_path)


del interface

interface = PPOInterface(training_generator, validation_generator, holdout_generator, task_predictor, img_shape, load_models=True, controller_save_path=controller_save_path, task_predictor_save_path = task_predictor_save_path)
###############################################################






# Hold-Out Evaluation
###############################################################
holdout_controller_preds = interface.get_controller_preds_on_holdout()



def reject_lowest_controller_valued_samples(rejection_ratio, holdout_controller_preds, holdout_generator):
    sorted_inds = np.argsort(holdout_controller_preds.reshape(len(holdout_controller_preds)))
    num_rejected = int(rejection_ratio * len(sorted_inds))
    
    selected_x_holdout = []
    selected_y_holdout = []
    for i in sorted_inds[num_rejected:]:
        X,y = holdout_generator[i]
        selected_x_holdout.append(X.reshape(36,36,1)) 
        selected_y_holdout.append(y[0])
    selected_x_holdout = np.array(selected_x_holdout)
    selected_y_holdout = np.array(selected_y_holdout)
    return selected_x_holdout, selected_y_holdout



def compute_mean_performance(x, y, interface):
    mean_performance_metric = interface.task_predictor.evaluate(x, y)
    return mean_performance_metric[-1]


performances = []
for rejection_ratio in np.arange(0.0, 0.5, 0.1):
    selected_x_holdout, selected_y_holdout = reject_lowest_controller_valued_samples(rejection_ratio, holdout_controller_preds, holdout_generator)
    performance = compute_mean_performance(selected_x_holdout, selected_y_holdout, interface)
    performances.append(performance)

reject_ratio = np.arange(0.0, 0.5, 0.1)
plt.plot(reject_ratio,performances)
plt.xlabel('Rejection Ratio')
plt.ylabel('Hold Out Accuracy')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig('PPO_Mnist_loader.pdf')
###############################################################

