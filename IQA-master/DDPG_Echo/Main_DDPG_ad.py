from Interface_DDPG_ad import DDPGInterface
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
data_dict_train = read_from_list(['database_path/train_labels_non.txt'],'')
data_dict_val = read_from_list(['database_path/valid_labels_non.txt'],'')
data_dict_holdout = read_from_list(['database_path/test_labels_non.txt'],'')

# Generators
training_generator_non = DataGenerator(data_dict_train, **params_train)
validation_generator_non = DataGenerator(data_dict_val, **params_val)

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

################################################################




#initial training on non expert training and validation data
###############################################################

task_predictor.fit_generator(generator=training_generator_non, use_multiprocessing=True, workers=6)
task_predictor.fit_generator(generator=validation_generator_non, use_multiprocessing=True, workers=6)

################################################################





# Adaptation on expert data
###############################################################

data_dict_train = read_from_list(['database_path/train_labels_2.txt'],'')
data_dict_val = read_from_list(['database_path/valid_labels_2.txt'],'')
data_dict_holdout = read_from_list(['database_path/test_labels_2.txt'],'')

# Generators
training_generator = DataGenerator(data_dict_train, **params_train)
validation_generator = DataGenerator(data_dict_val, **params_val)
holdout_generator = DataGenerator(data_dict_holdout, **params_holdout)
################################################################



#Controller 
###############################################################
num_train_episodes = 500



controller_batch_size = 512
task_predictor_batch_size = 256

interface = DDPGInterface(training_generator,validation_generator,holdout_generator, task_predictor, img_shape,
                          custom_controller=False,
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


###############################################################






#Holdout Eval
###############################################################


interface = DDPGInterface(training_generator,validation_generator,holdout_generator, task_predictor, img_shape, 
                          load_models=True, controller_weights_save_path=controller_weights_save_path, task_predictor_save_path=task_predictor_save_path,
                          custom_controller=False,
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
plt.savefig('DDPG_Echo_adapt.pdf')
###############################################################