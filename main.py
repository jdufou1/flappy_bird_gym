"""
Imports
"""
from xmlrpc.client import Boolean
from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from collections import deque
import time
from time import gmtime, strftime
import sys
import cv2
from utils import *
from torch.utils.tensorboard import SummaryWriter

"""
Script d'execution d'un D3QN pour l'environnement flappy bird gym
credit : https://github.com/Talendar/flappy-bird-gym
"""
"""
global variable
"""


path_q_network = "q_network_flappybird_weights"
path_q_target_network = "q_target_network_flappybird_weights"
path_best_model_network = "best_model_network_flappybird_weights"


class State(object) :

    def __init__(self,
        q_network,
        q_target_network,
        best_model,
        optim,
        nb_episode,
        discount_factor,
        learning_rate,
        test_frequency,
        nb_tests_iteration,
        epsilon_decay,
        epsilon_min,
        epsilon,
        batch_size,
        size_replay_buffer,
        update_frequency,
        tau,
        best_value,
        device
    ) : 
        self.q_network = q_network
        self.q_target_network = q_target_network
        self.best_model = best_model
        self.optim = optim
        self.nb_episode = nb_episode
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.test_frequency = test_frequency
        self.nb_tests_iteration = nb_tests_iteration
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.size_replay_buffer = size_replay_buffer
        self.update_frequency = update_frequency
        self.tau = tau
        self.best_value = best_value
        self.device = device
        self.replay_buffer = deque(maxlen=size_replay_buffer)
        self.current_episode = 0
        self.timestep = 0


"""
Dueling Deep Q Network
"""

class DuelingQNetwork(nn.Module) :
    """
    Implementation de la classe Dueling DQN
    """

    def __init__(self,
        nb_actions : int
    ) : 
        
        super().__init__()
        self.nb_actions = nb_actions 

        # Linear -> ReLU -> Linear -> ReLU -> Linear
        self.net =  nn.Sequential(
            nn.Conv2d(4,32,8,stride = 4,padding=(0,0)),
            nn.MaxPool2d(kernel_size=2 , padding=(0,0)),
            nn.ReLU(),
            nn.Conv2d(32,64,4,stride = 2 , padding = (1,1)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, stride = 2,padding = (1,1)),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride = 1, padding = (1,1)),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, padding = (1,1)),
            nn.Flatten(),
            nn.Linear(256, 256),
        )

        # implementation pour la fonction avantage 
        # correspond a la sortie avantage du réseau de neurones
        self.net_advantage = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, self.nb_actions)
        )

        # implementation pour la value function  
        # correspond a la sortie value function du réseau de neurones
        self.net_state_value = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256,1)
        )
        
    def advantage(self,x) :
        return self.net_advantage(self.net(x))
    
    def state_value(self,x) :
        return self.net_state_value(self.net(x))
    
    def forward(self,x) :
        return self.state_value(x) + self.advantage(x) - torch.mean(self.advantage(x),dim=1).unsqueeze(1)



class TrainModel :
    """
    Class for training our D3QN Model
    """
    def __init__(self,
        state : State
    ) :
        
        self.state = state # Etat du modèle

        self.writer = SummaryWriter("./logs/d3qn_flappy_bird_rgb_rewards")
        
        """
        TESTING
        """
        self.list_mean_rewards = list() 
        self.list_std_rewards = list()

        self.print_params()

        
        
    def pre_processing(self,state,old_frame = None) : 
        """
            input : np.array : 
            return : torch.tensor : 
        """
        if old_frame is None : 
            state_converted = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
            state_stacked = np.stack((state_converted, state_converted, state_converted, state_converted), axis=2)
            state_t = torch.as_tensor(state_stacked , device = self.state.device, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)
        else :
            new_state_converted = cv2.cvtColor(cv2.resize(state, (80, 80)), cv2.COLOR_BGR2GRAY)
            new_state_converted = new_state_converted.reshape(new_state_converted.shape[0],new_state_converted.shape[1],1 )
            state_stacked = np.append(new_state_converted, old_frame[:, :, :3], axis = 2)
            state_t = torch.as_tensor(state_stacked , device = self.state.device, dtype = torch.float32).unsqueeze(0).transpose(3 , 1)
            
        return state_stacked,state_t
        
        
    def training(self) :
        
        start_time = time.time()
        
        env = FlappyBirdEnvRGB()
        
        for episode in range(self.state.current_episode,self.state.nb_episode) :
            self.state.current_episode = episode

            state = env.reset()
            state,state_t = self.pre_processing(state)
            done = False
            cumul = 0 # reward total
            
            self.state.epsilon = max(self.state.epsilon * self.state.epsilon_decay,self.state.epsilon_min)
    
            while not done : 
                
                # epsilon-greedy pour la selection de l'action 
                if random.random() > self.state.epsilon : 
                    action = torch.argmax(self.state.q_network(state_t)).item()
                else :
                    action = env.action_space.sample()
            
                new_state,reward,done,_ = env.step(action)

                new_state_stacked,new_state_t = self.pre_processing(new_state,state)
                
                cumul += reward
                
                transition = (state_t,action,done,reward,new_state_t) # a ajouter dans le replay buffer
                self.state.replay_buffer.append(transition)

                if len(self.state.replay_buffer) >= self.state.batch_size and self.state.timestep % self.state.update_frequency == 0 :
                    
                    # selection du batch aleatoirement
                    batch = random.sample(self.state.replay_buffer,self.state.batch_size)

                    # transformation en tensor
                    
                    states = [exp[0] for exp in batch]
                    actions = np.asarray([exp[1] for exp in batch],dtype=int)
                    dones = np.asarray([exp[2] for exp in batch],dtype=int)
                    rewards = np.asarray([exp[3] for exp in batch],dtype=np.float32)
                    new_states = [exp[4] for exp in batch]
                    

                    states_t = torch.stack(states).squeeze(1)
                    new_states_t = torch.stack(new_states).squeeze(1)

                    dones_t = torch.as_tensor(dones , dtype = torch.int64, device = self.state.device).unsqueeze(1)
                    actions_t = torch.as_tensor(actions , dtype = torch.int64, device = self.state.device).unsqueeze(1)
                    rewards_t = torch.as_tensor(rewards , dtype=torch.float32, device = self.state.device).unsqueeze(1)

                    # l'esperance des futurs rewards
                    y_target = (rewards_t + 
                                self.state.discount_factor * 
                                (1 - dones_t) * 
                                torch.gather(self.state.q_target_network(new_states_t),dim=1,index=torch.argmax(self.state.q_network(new_states_t),dim=1).unsqueeze(1)).detach())

                    # descente de gradient 
                    mse = nn.MSELoss()
                    loss = mse(torch.gather(self.state.q_network(states_t),dim=1,index=actions_t), y_target)
                    self.state.optim.zero_grad()
                    loss.backward()
                    self.state.optim.step()
                    self.update_q_target_network()

                self.state.timestep += 1
                state = new_state_stacked
                state_t = new_state_t
            
            # Partie Test
            if episode % self.state.test_frequency == 0 :
                
                mean_rewards, std_rewards = self.testing()
                
                self.list_mean_rewards.append( mean_rewards )
                self.list_std_rewards.append( std_rewards )
                
                end_time = time.time()
                diff_time = end_time - start_time
                start_time = time.time()
                """
                Keep the best model
                """
                if mean_rewards > self.state.best_value :
                    self.state.best_value = mean_rewards
                    self.state.best_model.load_state_dict(self.state.q_network.state_dict())
                
                """
                Saving
                """
                self.save_model()
                # self.write_params(mean_rewards, std_rewards)

                """
                display :
                """
                self.writer.add_scalar('TP3 : rewards FlappyBird', mean_rewards, episode)
                print(f"[LOG] : ({(episode/self.state.nb_episode*100)}%) - Episode : {episode} - mean rewards : {mean_rewards} - std rewards : {std_rewards} - eps : {self.state.epsilon} - time : {diff_time}s - best value : {self.state.best_value} - rb : {len(self.state.replay_buffer)}")
            
    def update_q_target_network(self) : 
        """ update du q-target en fonction du tau"""   
        for target_param, local_param in zip(self.state.q_target_network.parameters(), self.state.q_network.parameters()):
            target_param.data.copy_(self.state.tau*local_param.data + (1.0-self.state.tau)*target_param.data)          
        
    def testing(self) :
        """
            return :
                list_cum_sum.mean : float
                list_cum_sum.std : float
        """
        list_cum_sum = list()
        for _ in range(self.state.nb_tests_iteration) :
            list_cum_sum.append(self.one_test_model())
            
        list_cum_sum = np.asarray(list_cum_sum , dtype=np.float32)    
        
        return list_cum_sum.mean(), list_cum_sum.std()
            
    def one_test_model(self) :
        """
            return :
                cum_sum : float (reward of an episode with the current q_network)
        """
        random.seed(10)
        
        env = FlappyBirdEnvRGB()
        
        state = env.reset()
        state,state_t = self.pre_processing(state)
        done = False
        cum_sum = 0.0
        while not done :
            action = torch.argmax(self.state.q_network(state_t)).item()
            new_state,reward,done,_ = env.step(action)
            new_state_stacked,new_state_t = self.pre_processing(new_state,state)
            
            state = new_state_stacked
            state_t = new_state_t
            cum_sum += reward
                
        return cum_sum
    
    def save_model(self) :
        try : 
            savepath = Path("./parameters_flappy_bird_d3qn.pch")
            with savepath.open("wb") as fp :
                torch.save(self.state,fp)
            print("[STATUS] : Models saving... SUCCESS")
        except KeyboardInterrupt :    
            print("[STATUS] : Models saving... FAIL KeyboardInterrupt")
        except Exception as e :
            print("[STATUS] : Models saving... FAIL ",e)



    def print_params(self) :
        print("[LOG] : Params : ")
        print("1#      current_episode : "+str(self.state.current_episode))
        print("2#      nb_episode : "+str(self.state.nb_episode))
        print("3#      discount_factor : "+str(self.state.discount_factor))
        print("4#      learning_rate : "+str(self.state.learning_rate))
        print("5#      current_episode : "+str(self.state.current_episode))
        print("6#      test_frequency : "+str(self.state.test_frequency))
        print("7#      nb_tests_iteration : "+str(self.state.nb_tests_iteration))
        print("8#      epsilon_decay : "+str(self.state.epsilon_decay))
        print("9#      epsilon_min : "+str(self.state.epsilon_min))
        print("10#     epsilon : "+str(self.state.epsilon))
        print("11#     batch_size : "+str(self.state.batch_size))
        print("12#     size_replay_buffer : "+str(self.state.size_replay_buffer))
        print("13#     update_frequency : "+str(self.state.update_frequency))
        print("14#     tau : "+str(self.state.tau))
        print("15#     best_value : "+str(self.state.best_value))


if __name__ == '__main__':


    print("##################################")
    print("      D3QN for Flappy Bird        ")
    print("- Target and Replay Buffer        ")
    print("- Dueling Network                 ")
    print("- Double DQNetwork                ")
    print("##################################")


    args = sys.argv[1:]

    restart = False
    if len(args) > 0 :
        restart = True


    if restart :
        print("[LOG] : Restart mode selected")
        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LOG] : Device used : {device}")
        # params = get_dict_last_params("./csv_data_parameters_flappy_bird.csv")

        savepath = Path("./parameters_flappy_bird_d3qn.pch")
        if savepath.is_file() :
            with savepath.open("rb") as fp :
                state = torch.load(fp)
        else :
            print("[ERROR] : Restart didn't work : savepath ins't a file")
       
    else :
        print("[LOG] : Initialisation mode selected")
        device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LOG] : Device used : {device}")
        learning_rate = 0.0003
        q_network = DuelingQNetwork(
            nb_actions = 2
        ).to(device)
        
        q_target_network = DuelingQNetwork(
            nb_actions = 2
        ).to(device)
        
        best_model = DuelingQNetwork(
            nb_actions = 2
        ).to(device)
        
        """ q_network copy """
        q_target_network.load_state_dict(q_network.state_dict())
        best_model.load_state_dict(q_network.state_dict())

        # on utilise comme optimizer Adam (a tester avec d'autres) 
        optim = torch.optim.Adam(q_network.parameters(), lr=learning_rate) 

        state = State(
            q_network,
            q_target_network,
            best_model,
            optim,
            nb_episode = 10000000,
            discount_factor = 0.99,
            learning_rate = learning_rate,
            update_frequency = 1,
            nb_tests_iteration = 10,
            epsilon_decay = 0.9995,
            epsilon_min = 0.02,
            epsilon = 1.0,
            batch_size = 64,
            size_replay_buffer = 50000,
            test_frequency = 10,
            tau = 1e-3,
            best_value = -1e10,
            device = device
        )
        
    train_model = TrainModel(state)

    train_model.training()