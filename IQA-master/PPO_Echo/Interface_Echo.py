from Task_Amenability_Echo import TaskAmenability
import numpy as np
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise




class PPOInterface():
    def __init__(self,training_generator, validation_generator, holdout_generator, task_predictor, img_shape, load_models=False, controller_save_path=None, task_predictor_save_path=None):
        
        self.holdout_generator = holdout_generator
        
        if load_models:
            self.task_predictor = load_model(task_predictor_save_path)
        else:
            self.task_predictor = task_predictor
        
        def make_env():
            return TaskAmenability(training_generator, validation_generator, holdout_generator, task_predictor, img_shape)


        self.env = DummyVecEnv([make_env])

        def get_from_env(env, parameter):
            return env.get_attr(parameter)[0]

        self.n_rollout_steps = get_from_env(self.env, 'controller_batch_size') + get_from_env(self.env, 'num_val')# number of steps per episode (controller_batch_size + val_set_len) multiply by an integer to do multiple episodes before controller update
        
        if load_models:
            assert isinstance(controller_save_path, str)
            self.load(save_path=controller_save_path)
        else:
            self.model = PPO('CnnPolicy', 
                              self.env,
                              batch_size=3,
                              n_steps=self.n_rollout_steps,
                              gamma=0.98,
                              verbose=2,
                              seed=None
                              )

    def train(self, num_episodes):
        time_steps = int(num_episodes*self.n_rollout_steps)
        
        print(f'Training started for {num_episodes} episodes:')
        
        self.model.learn(total_timesteps=time_steps)
        
    def get_controller_preds_on_holdout(self):
        actions = []
        for i in range(len(self.holdout_generator)):
            pred = self.model.predict(self.holdout_generator[i][0])[0][0]
            actions.append(pred)
        return np.array(actions)

    def save(self, controller_save_path, task_predictor_save_path):
        self.model.save(controller_save_path)
        task_predictor_copy = self.env.get_attr('task_predictor')[0]
        task_predictor_copy.save(task_predictor_save_path)
        
    def load(self, save_path):
        self.model = PPO.load(save_path)
        self.model.set_env(self.env)
        
        

        
 
        
