from Interface_DDPG import DDPGInterface
from dataloader import DataGenerator
from tensorflow.keras import layers
from tensorflow.keras import layers
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
from List_Read import read_from_list
from tensorflow.keras import regularizers




img_shape = (224, 224, 1)


# GPU
###############################################################
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

view_classes = ['AP2', 'AP3', 'AP4', 'AP5', 'PLAX', 'RVIF', 'SUBC4', #'SUBC5'
            'SUBIVC', 'PSAX(A)', 'PSAX(M)', 'PSAX(PM)',  'PSAX(APIX)', 'SUPRA'] # 'unknown', 'garbage]
###############################################################




#DataLoader for ECHO
##############################################################

params_train = {'dim': (224,224),
          'batch_size': 512,
          'n_classes': 14,
          'n_channels': 1,
          'shuffle': False}

# Parameters_validation
params_val = {'dim': (224,224),
          'batch_size': 1,
          'n_classes': 14,
          'n_channels': 1,
          'shuffle': False}

params_holdout = {'dim': (224,224),
          'batch_size': 1,
          'n_classes': 14,
          'n_channels': 1,
          'shuffle': True}
# Datasets
data_dict_train = read_from_list(['database_path/train_labels_2.txt'],'')
data_dict_val = read_from_list(['database_path/valid_labels_2.txt'],'')
data_dict_holdout = read_from_list(['database_path/test_labels_2.txt'],'')

# Generators
training_generator = DataGenerator(data_dict_train, **params_train)
validation_generator = DataGenerator(data_dict_val, **params_val)
holdout_generator = DataGenerator(data_dict_holdout, **params_holdout)

##############################################################





#taskpredictor Mobilenet
###############################################################

base_model = keras.applications.MobileNetV2(
    weights=None,  
    input_shape=img_shape,
    include_top=False) 
inputs = keras.Input(shape=img_shape)
x = base_model(inputs, training=True)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(14,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
task_predictor=keras.Model(inputs, outputs)
 
task_predictor.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[tensorflow.keras.metrics.CategoricalAccuracy()])

# ###############################################################






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
    act_x = layers.Conv2D(128, (3,3), activation='relu')(act_x)
    act_x = layers.MaxPool2D((2,2))(act_x)
    act_x = layers.Conv2D(256, (3,3), activation='relu')(act_x)
    act_x = layers.MaxPool2D((2,2))(act_x)
    act_x = layers.Conv2D(256, (3,3), activation='relu')(act_x)
    act_x = layers.Flatten()(act_x)
    act_x = layers.Dense(1024, activation='relu')(act_x)
    act_x = layers.Dense(256, activation='relu')(act_x)
    act_x = layers.Dense(64, activation='relu')(act_x)
    act_out = layers.Dense(n_actions, activation='sigmoid')(act_x)
    actor = keras.Model(inputs=act_in, outputs=act_out)
    
    action_input = layers.Input(shape=(n_actions,), name='action_input')
    observation_input = layers.Input((1,) + img_shape, name='observation_input')
    observation_input_reshape = layers.Reshape((img_shape))(observation_input)
    observation_x = layers.Conv2D(32, (3,3), activation='relu')(observation_input_reshape)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(64, (3,3), activation='relu')(observation_x)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(128, (3,3), activation='relu')(observation_x)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(256, (3,3), activation='relu')(observation_x)
    observation_x = layers.MaxPool2D((2,2))(observation_x)
    observation_x = layers.Conv2D(256, (3,3), activation='relu')(observation_x)
    flattened_observation = layers.Flatten()(observation_x)
    x = layers.Concatenate()([action_input, flattened_observation])
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1)(x)
    critic = keras.Model(inputs=[action_input, observation_input], outputs=x)
    return actor, critic, action_input


actor, critic, action_input = build_actor_critic(img_shape)

controller_batch_size = 512
task_predictor_batch_size = 256

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

save_path = r'temp/echo_experiment_train_session'

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
        selected_x_holdout.append(X.reshape(224,224,1)) 
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
plt.savefig('DDPG_ECHO.pdf')
###############################################################