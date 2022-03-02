import gym
import numpy as np
from gym import spaces
from tensorflow.keras import layers
from tensorflow import keras 



class TaskAmenability(gym.Env):
    
    def __init__(self, training_generator, validation_generator, holdout_generator, task_predictor, img_shape):

        self.training_generator = training_generator
        self.validation_generator = validation_generator
        self.holdout_generator = holdout_generator

        self.img_shape = img_shape
        
        self.task_predictor = task_predictor
        
        self.controller_batch_size = 64
        self.task_predictor_batch_size = 32
        self.epochs_per_batch = 2
        
        self.img_shape = img_shape
        
        self.num_val = len(self.validation_generator)
        
     
        self.observation_space =  spaces.Box(low=0, high=255, shape=self.img_shape, dtype=np.uint8)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.actions_list = []
        self.val_metric_list = [0.5]*10
        
        self.sample_num_count = 0
        self.b_indx = 0
        self.o_index =0 
        

    def compute_moving_avg(self):
        self.val_metric_list = self.val_metric_list[-10:]
        moving_avg = np.mean(self.val_metric_list)
        return moving_avg
    
    def select_samples(self, actions_list):
        actions_list = np.clip(actions_list, 0, 1)
        selection_vector = np.random.binomial(1, actions_list)
        logical_inds = [bool(elem) for elem in selection_vector]
        return self.x_train_batch[logical_inds], self.y_train_batch[logical_inds]
    
    def get_val_acc_vec(self):
        val_acc_vec = []
        for i in range(self.num_val):
            metrics = self.task_predictor.evaluate(self.validation_generator[i][0], self.validation_generator[i][1], verbose=0)
            val_metric = metrics[-1]
            val_acc_vec.append(val_metric)
        #print('get_val_acc_vec')
        return np.array(val_acc_vec)
    
    def step(self, action):
        self.actions_list.append(action)
        self.sample_num_count += 1

        if self.sample_num_count < self.controller_batch_size+self.num_val:
            reward = 0
            done = False

            if self.o_index < self.controller_batch_size-1:
                self.o_index += 1
                X = self.x_train_batch[self.o_index]
                return X , reward, done, {}
            else:
                return np.squeeze(self.validation_generator[self.sample_num_count-self.o_index-1][0],axis=0), reward, done, {}

        
        else:
            x_train_selected, y_train_selected = self.select_samples(self.actions_list[:self.controller_batch_size])
            print('x_train_selected shape', x_train_selected.shape)
            print('y_train_selected shape', y_train_selected.shape)
            if len(y_train_selected) < 1:
                reward = -1
                done = True
            else:
                moving_avg = self.compute_moving_avg()

                self.task_predictor.fit(x_train_selected, y_train_selected, batch_size=self.task_predictor_batch_size, epochs=self.epochs_per_batch, shuffle=True, verbose=0)

                val_acc_vec = self.get_val_acc_vec()
                val_sel_vec = self.actions_list[self.controller_batch_size:]
                val_sel_vec_normalised = np.array(val_sel_vec) / np.mean(val_sel_vec)

                val_metric = np.mean(np.multiply(val_sel_vec_normalised, np.array(val_acc_vec)))
                
                self.val_metric_list.append(val_metric)
                reward = val_metric - moving_avg
                done = True

            return np.random.rand(self.img_shape[0], self.img_shape[1], self.img_shape[2]), reward, done, {}
        
    def reset(self):
        if self.b_indx == len(self.training_generator):
            self.b_indx=0
        self.x_train_batch, self.y_train_batch = self.training_generator[self.b_indx][0], self.training_generator[self.b_indx][1]
        self.b_indx = self.b_indx + 1 
        self.actions_list = []
        self.sample_num_count = 0
        self.o_index =0 

        return self.x_train_batch[self.sample_num_count]
    
    def save_task_predictor(self, task_predictor_save_path):
        self.task_predictor.save(task_predictor_save_path)
