from Interface_DDPG import DDPGInterface
from dataloader_mnist import DataGenerator
from tensorflow import keras
import numpy as np
import os
import tensorflow
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import regularizers



img_shape = (36, 36, 1)



#Data Generator 
###############################################################

params_train = {'dim': (36,36),
          'batch_size': 64,
          'n_classes':  2,
          'n_channels': 1,
          'shuffle': False}

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





#Controller 
###############################################################
num_train_episodes = 500

def build_actor_critic(img_shape, action_shape=(1,)):
    
    n_actions = action_shape[0]
    
    act_in = layers.Input((1,) + img_shape)
    act_in_reshape = layers.Reshape((img_shape))(act_in)
    act_x = layers.Conv2D(32, (3,3), activation='relu')(act_in_reshape)
    act_x = layers.MaxPool2D((2,2))(act_x)
    act_x = layers.Conv2D(64, (3,3), activation='relu')(act_x)
    act_x = layers.MaxPool2D((2,2))(act_x)
    act_x = layers.Conv2D(64, (3,3), activation='relu')(act_x)
    act_x = layers.Flatten()(act_x)
    act_x = layers.Dense(64, activation='relu')(act_x)
    act_x = layers.Dense(32, activation='relu')(act_x)
    act_x = layers.Dense(16, activation='relu')(act_x)
    act_out = layers.Dense(n_actions, activation='sigmoid')(act_x)
    actor = keras.Model(inputs=act_in, outputs=act_out)
    
    action_input = layers.Input(shape=(n_actions,), name='action_input')
    observation_input = layers.Input((1,) + img_shape, name='observation_input')
    observation_input_reshape = layers.Reshape((img_shape))(observation_input)
    observation_x = layers.Conv2D(32, (3,3), activation='relu')(observation_input_reshape)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(64, (3,3), activation='relu')(observation_x)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(64, (3,3), activation='relu')(observation_x)
    flattened_observation = layers.Flatten()(observation_x)
    x = layers.Concatenate()([action_input, flattened_observation])
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(1)(x)
    critic = keras.Model(inputs=[action_input, observation_input], outputs=x)
    return actor, critic, action_input


actor, critic, action_input = build_actor_critic(img_shape)

controller_batch_size = 64
task_predictor_batch_size = 32

interface = DDPGInterface(training_generator,validation_generator,holdout_generator, task_predictor, img_shape,
                          custom_controller=True, actor=actor, critic=critic, action_input=action_input,
                          modify_env_params=True, modified_env_params_list=[controller_batch_size, task_predictor_batch_size])


interface.train(num_train_episodes)
###############################################################







#Save 
###############################################################
save_dir = 'temp'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

save_path = r'temp/pneumoniamnist_experiment_train_session'

controller_weights_save_path = save_path + 'controller_episode_' + str(num_train_episodes)
task_predictor_save_path = save_path + 'task_predictor_episode_' + str(num_train_episodes)

interface.save(controller_weights_save_path=controller_weights_save_path,
               task_predictor_save_path=task_predictor_save_path)

del interface
del actor 
del critic
del action_input
###############################################################






#Holdout Eval
###############################################################
actor, critic, action_input = build_actor_critic(img_shape)


interface = DDPGInterface(training_generator,validation_generator,holdout_generator, task_predictor, img_shape, 
                          load_models=True, controller_weights_save_path=controller_weights_save_path, task_predictor_save_path=task_predictor_save_path,
                          custom_controller=True, actor=actor, critic=critic, action_input=action_input,
                          modify_env_params=True, modified_env_params_list=[controller_batch_size, task_predictor_batch_size])


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
plt.savefig('DDPG_Mnist.pdf')
###############################################################
