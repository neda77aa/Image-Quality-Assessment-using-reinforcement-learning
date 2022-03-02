from Task_Amenability_DDPG_ad import TaskAmenability
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.optimizers import Adam 
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Merge, Activation, Embedding
from keras.layers.recurrent import LSTM

class DDPGInterface():
    def __init__(self, training_generator, validation_generator, holdout_generator, task_predictor, img_shape, load_models=False, controller_weights_save_path=None, task_predictor_save_path=None, custom_controller=False, actor=None, critic=None, action_input=None, modify_env_params=False, modified_env_params_list=None):
        
        self.holdout_generator = holdout_generator
        
        if load_models:
            self.task_predictor = load_model(task_predictor_save_path)
        else:
            self.task_predictor = task_predictor
        
        self.env = TaskAmenability(training_generator,validation_generator,holdout_generator, task_predictor, img_shape)

        self.n_rollout_steps = self.env.controller_batch_size + self.env.num_val # number of steps per episode (controller_batch_size + val_set_len) multiply by an integer to do multiple episodes before controller update
        
        n_actions = self.env.action_space.shape[0]
        
        if not custom_controller:
            
            action_input = layers.Input(shape=(n_actions,), name='action_input')
            observation_input = layers.Input((1,) + img_shape, name='observation_input')
            
            act_in = layers.Input((1,) + self.env.observation_space.shape)
            act_in_reshape = layers.Reshape((self.env.img_shape))(act_in)
            

            x = TimeDistributed(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu'))(act_in_reshape)
            x = TimeDistributed(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))(x)
            x = TimeDistributed(Convolution2D(64, 3, 3, activation='relu'))(x)
            x = TimeDistributed(Flatten())(x)

            x = LSTM(512, activation='tanh')(x)     
            
            actor0 = Dense(n_actions, activation='softmax')(x)
            critic0 = Dense(1, activation='linear')(x)
        
            actor0 = keras.Model(inputs=act_in, outputs=actor0)
            critic0  = keras.Model(inputs=[action_input, observation_input], outputs=critic0)
            
        

            
        else:
            pass
        
        memory_limit = int(6*self.n_rollout_steps)
        warmup_steps = int(2*self.n_rollout_steps)
        train_interval = int(2*self.n_rollout_steps)
        
        memory = SequentialMemory(limit=memory_limit, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=n_actions, theta=.15, mu=0., sigma=.3)
        
        self.agent = DDPGAgent(nb_actions=n_actions, actor=actor, critic=critic, critic_action_input=action_input,
                               memory=memory, nb_steps_warmup_critic=warmup_steps, nb_steps_warmup_actor=warmup_steps,
                               random_process=random_process, gamma=1, target_model_update=1e-3, batch_size=self.env.controller_batch_size, 
                               train_interval=train_interval)
        
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mse'])

        if modify_env_params:
            controller_batch_size = modified_env_params_list[0]
            task_predictor_batch_size = modified_env_params_list[1]
            self.env.controller_batch_size = controller_batch_size
            self.env.task_predictor_batch_size = task_predictor_batch_size
        
        if load_models:
            self.train(1)
            self.agent.load_weights(controller_weights_save_path)
            self.task_predictor = load_model(task_predictor_save_path)
                        
    def train(self, num_episodes):
        for i in range(num_episodes):
            self.agent.fit(self.env, int(self.n_rollout_steps))
        
    def get_controller_preds_on_holdout(self):
        actions = []
        for i in range(len(self.holdout_generator)):
            pred = self.agent.actor.predict(np.expand_dims(self.holdout_generator[i][0], axis=0))[0][0]
            actions.append(pred)

        return np.array(actions)
        
    def save(self, controller_weights_save_path, task_predictor_save_path):
        self.agent.save_weights(controller_weights_save_path, overwrite=True)
        self.env.save_task_predictor(task_predictor_save_path)