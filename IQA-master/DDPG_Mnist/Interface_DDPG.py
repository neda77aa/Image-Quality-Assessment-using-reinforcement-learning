from Task_Amenability_DDPG import TaskAmenability
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
            act_in = layers.Input((1,) + self.env.observation_space.shape)
            act_in_reshape = layers.Reshape((self.env.img_shape))(act_in)
            act_x = layers.Conv2D(32, (3,3), activation='relu')(act_in_reshape)
            act_x = layers.MaxPool2D((2,2))(act_x)
            act_x = layers.Conv2D(32, (3,3), activation='relu')(act_x)
            act_x = layers.MaxPool2D((2,2))(act_x)
            act_x = layers.Flatten()(act_x)
            act_x = layers.Dense(64, activation='relu')(act_x)
            act_x = layers.Dense(32, activation='relu')(act_x)
            act_x = layers.Dense(16, activation='relu')(act_x)
            act_out = layers.Dense(n_actions, activation='sigmoid')(act_x)
            actor = keras.Model(inputs=act_in, outputs=act_out)
            
            action_input = layers.Input(shape=(n_actions,), name='action_input')
            observation_input = layers.Input((1,) + self.env.observation_space.shape, name='observation_input')
            observation_input_reshape = layers.Reshape((self.env.img_shape))(observation_input)
            observation_x = layers.Conv2D(32, (3,3), activation='relu')(observation_input_reshape)
            observation_x = layers.MaxPool2D((2,2))(observation_x)
            observation_x = layers.Conv2D(16, (3,3), activation='relu')(observation_x)
            observation_x = layers.MaxPool2D((2,2))(observation_x)
            flattened_observation = layers.Flatten()(observation_x)
            x = layers.Concatenate()([action_input, flattened_observation])
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dense(16, activation='relu')(x)
            x = layers.Dense(1)(x)
            critic = keras.Model(inputs=[action_input, observation_input], outputs=x)
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
        
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

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
        self.agent.fit(self.env, int(num_episodes*self.n_rollout_steps))
        
    def get_controller_preds_on_holdout(self):
        actions = []
        for i in range(len(self.holdout_generator)):
            pred = self.agent.actor.predict(np.expand_dims(self.holdout_generator[i][0], axis=0))[0][0]
            actions.append(pred)

        return np.array(actions)
        
    def save(self, controller_weights_save_path, task_predictor_save_path):
        self.agent.save_weights(controller_weights_save_path, overwrite=True)
        self.env.save_task_predictor(task_predictor_save_path)