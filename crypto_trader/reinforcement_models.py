# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:09:13 2021

@author: Joe
"""
import os
import itertools
import random
import copy
from abc import ABC, abstractmethod
from collections import namedtuple

import pickle
import torch
import torch.optim as optim
import torch.nn.functional as functional
import torch.nn as nn

import numpy as np
import pandas as pd

from crypto_trader.simulated_trader import simulated_bot_action_deterministic, get_base_balance
from crypto_trader.cnn_models import CustomCommonLayersCNN
from crypto_trader.data import get_stream_data_sizes, prep_sim_data

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
        
class UnsupervisedLearning(ABC):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def optimize_model():
        pass
    
    @abstractmethod
    def training_loop():
        pass    

    @abstractmethod
    def get_action(state):
        pass
    
    
class DeepQLeaning(UnsupervisedLearning):

    def __init__(
            self,
            memory_size,
            ticker='XRPEUR',
            data_path='data/streams/XRPEUR/',
            play_path='data/streams/XRPEUR/random_play',
            create_torch_models=True
        ):
        super().__init__()
        
        self.play_path = play_path
        
        self.ticker = ticker
        self.memory = self.ReplayMemory(memory_size, play_path)
        self.data_path = data_path
        
        self.balance = {'EUR': 1000, 'XRP': 0}
        
        (num_numeric_features,
         self.image_size,
         self.stream_dates,
         self.stream_times) = get_stream_data_sizes(self.data_path, half=True)
        
        #print(self.stream_dates)
        
        self.num_actions = 3 # buy/sell/doNothing, amount 
        self.scale = 3 # 1.5
        self.batch_size = 32
        self.gamma = 0.99995 # weight for future rewards, could be 25000 steps before getting reward (sell)

        self.episode_durations = []
        self.terminating_balance = 100
        #print(num_numeric_features)
        self.num_numeric_features = 15 #num_numeric_features + len(self.balance)
        
        self.delayed_state_adder = self.DelayedStateAdder()
            
        self.policy_net, self.target_net = None, None
        if create_torch_models:
            self.policy_net = CustomCommonLayersCNN(
                num_numerical_inputs=self.num_numeric_features,
                image_size=self.image_size,
                num_outputs=self.num_actions,
                scale=self.scale
            ).to(self.device)
            
            self.target_net = CustomCommonLayersCNN(
                num_numerical_inputs=self.num_numeric_features,
                image_size=self.image_size,
                num_outputs=self.num_actions,
                scale=self.scale
            ).to(self.device)
            
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        
            self.optimizers = [
                 optim.Adadelta(self.policy_net.parameters(), lr=0.001),
                
                 optim.SparseAdam(self.policy_net.parameters(), lr=0.001),
                 optim.Adam(self.policy_net.parameters(), lr=0.001), ##!!
                 optim.ASGD(self.policy_net.parameters(), lr=0.001),
                 optim.RMSprop(self.policy_net.parameters(), lr=0.001),
                 optim.Adagrad(self.policy_net.parameters(), lr=0.001),
                 
                 optim.SparseAdam(self.policy_net.parameters(), lr=0.0001),
                 optim.Adam(self.policy_net.parameters(), lr=0.00001),
                 optim.ASGD(self.policy_net.parameters(), lr=0.0001),
                 optim.RMSprop(self.policy_net.parameters(), lr=0.0001),
                 optim.Adagrad(self.policy_net.parameters(), lr=0.0001),
                 
            ]
            self.optimizer = self.optimizers[-3] # 3
            
            self.activation_functions = [
                functional.selu, # -1.7 to inf smooth
                functional.softsign, # -1 to 1 v smooth
                functional.relu, # 0 to inf hard gate
                functional.relu6, # 0 to 6 hard gate
                functional.hardswish, # relu but quadratic near 0 
                functional.hardtanh, # -1 to 1, harsh/gate
                functional.rrelu, # gate, random leaky (more generalizable)
                functional.softplus, # all positive smooth transition
                functional.tanhshrink, # -inf -> inf & squashed near 0
                functional.logsigmoid, # all negative smooth transition
            ]
            self.activation_function =  self.activation_functions[-5] # 3
            
            self.loss_functions = [
                functional.kl_div,
                functional.smooth_l1_loss,
                functional.l1_loss,
                functional.mse_loss
            ]
            self.loss_function = self.loss_functions[-1] # 3
        
        
    def prepare_step_info_for_model(self, step_dict, normalizer=100000):
        current_step = step_dict['current_step'] if step_dict['current_step'] != 0 else 1
        return [
            (current_step - step)/normalizer
                for step_type, step in step_dict.items() if step_type != 'current_step'
        ]
        
    def prepare_state(self, state, step_dict, balance_denom=1000):
        numeric_input = np.array(
            np.concatenate([
                state[0],
                np.array(list(self.balance.values()))/balance_denom,
                self.prepare_step_info_for_model(step_dict),
            ], axis=0)
        ).astype(np.float16)
        
        image_array = np.moveaxis(state[1], -1, 0).astype('uint8')
        
        return [
            numeric_input,
            image_array
        ]
    
    
    def prefill_memory(self):
        for dataset in os.listdir(self.play_path):
            self.memory.memory.extend(pickle.load(open(
                self.play_path+dataset,
                'rb'
            )))     


    def perf_sim_step(
        self,
        current_step,
        sim_numeric_data,
        sim_image_data,
        sim_prices,
        buy_rewards,
        step_dict,
        random_settings,
        epsilon,
        next_step_offset=10,
        sim_action_rewards=None
    ):
        step_dict['current_step'] = current_step
        sim_finished = False 

        state = self.prepare_state(
            [sim_numeric_data[current_step], sim_image_data[current_step]],
            step_dict # (current_step-last_buy_step)/10000 #if last_buy_step > 0 else 0
        )

        current_price, next_price = sim_prices[current_step], sim_prices[current_step+1]
        
        # Get NN to select next action
        action, selected_by_network, cur_weights, fc_input = \
            self.get_action(state, epsilon=epsilon, r=random_settings.r)
        
        # perform action in sim env & get actual reward
        self.balance, skip_next, reward, base_balance_at_last_buy, performed_action = \
            simulated_bot_action_deterministic(
                self.balance,
                step_dict,
                action,
                next_price, # fed into neural net - part of state
                self.ticker,
                buy_rewards[current_step]
                #should_do_nothing_reward=-0.1
            )
        
        q, action_selected_by_network = torch.max(action, dim=1)
        
        if sim_action_rewards:
            sim_action_rewards = sim_action_rewards.append({
                "action": performed_action,
                "action_selected_by_network": action_selected_by_network,
                "selected_by_network": selected_by_network,
                "selected_pred_future_rewards": q[0].item(),
                "pred_future_rewards":action,
                "reward": reward,
                "step_buy_reward": buy_rewards[current_step],
                "weightedAvgPrice": weightedAvgPrice[current_step],
                "current_price": current_price,
                "cur_weights": cur_weights,
                "fc_input": fc_input,
            }, ignore_index=True)

        if action_selected_by_network == 0:
            step_dict['last_buy_atempt_step'] = current_step

        elif action_selected_by_network == 1:
            step_dict['last_sell_attempt_step'] = current_step
        
        # convert reward to tensor
        reward = torch.tensor([reward], device=self.device)
        
        base_balance = get_base_balance(self.balance, next_price, self.ticker)

        if base_balance < self.terminating_balance:
            print('\t\t Exceeded min balance...')
            sim_finished = True
        
        next_state = self.prepare_state(
            [sim_numeric_data[current_step + next_step_offset],
                sim_image_data[current_step + next_step_offset]],
            step_dict #(current_step-last_buy_step)/10000 #if last_buy_step > 0 else 0
        )
        
        return (state, action, next_state, reward), sim_finished, reward, performed_action

    def training_loop(
            self,
            random_settings,
            num_episodes=50,
            sim_indexes=None,
            optimize=True,
            next_step_offset=10,
        ):
        
        episode_start_balance = {'EUR': 1000, 'XRP': 0}
        
        #step_losses = []
        sim_num = 0
        
        self.pretraining(num_epochs=3, epoch_length=500, evaluate=False, include_zero=True)
        #self.prefill_memory()
        #self.memory.add_large_reward_states_to_memory()
        
        print('Mem Size', len(self.memory.memory))
        
        for i_episode in range(num_episodes):
            #epsilon = start_epsilon - (i_episode/num_episodes)            
    
            sim_indexes = sim_indexes if sim_indexes else range(len(self.stream_dates))
            
            print('Episode {}'.format(i_episode))
            
            for sim_index in sim_indexes:
            
                epsilon = random_settings.get_epsilon(sim_num)
                
                self.balance = copy.deepcopy(episode_start_balance)

                total_loss, skip_next = 0, False

                step_dict = {
                    'last_buy_atempt_step': 0,
                    'last_sell_attempt_step': 0,
                    'current_step': 0,
                    'base_balance_at_last_buy': self.balance['EUR']
                }

                sim_numeric_data, sim_image_data, buy_rewards, sim_prices, weightedAvgPrice = \
                    prep_sim_data(sim_index, self.stream_dates, self.stream_times, self.data_path)
                    
                print('\t Simulation={}, epsilon={}, length={}'.format(
                    sim_index, epsilon, sim_numeric_data.shape[0]
                ))
        
                max_step = sim_numeric_data.shape[0]-next_step_offset
                
                sim_action_rewards = pd.DataFrame()
                
                for current_step in range(max_step):

                    # add transaction delay
                    if skip_next:
                        skip_next = False
                    else: 
                        state_tuple, sim_finished, reward, _ = self.perf_sim_step(
                            current_step,
                            sim_numeric_data,
                            sim_image_data,
                            sim_prices,
                            buy_rewards,
                            step_dict,
                            random_settings,
            	            epsilon,
                            next_step_offset,
                            sim_action_rewards
                        )
                        
                        self.delayed_state_adder.add_or_delay_state(
                            current_step,
                            state_tuple,
                            self.memory
                        )
                        
                        if optimize and current_step % int(self.batch_size/4) == 0:
                            loss = self.optimize_model()
                    
                            total_loss += loss if loss else 0
                            #step_losses.append(loss)
                        
                            if current_step % 10000 == 0:
                                print(
                                    '\t\t Step={}, Loss={},\n\t\t\t Total Loss={}, Balance={},'
                                    '\n\t\t\tlatest ActionQVal={}, ratio={}'.format(
                                        current_step, loss,
                                        total_loss, self.balance,
                                        action[0], self.memory.stored_ratio
                                    )
                                )  

                        if sim_finished:
                            break

                sim_action_rewards.to_csv('sim_action_rewards.csv', index=False)

                self.episode_durations.append(current_step)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
                print(
                    '\t\t Finishing sim {}. Start balance={},\n\t\t finishing balance={}\n\t\t final_step={}, max_step={}'.format(
                        sim_index, episode_start_balance, self.balance, current_step, sim_numeric_data.shape[0]
                    )
                )
                
                #self.pretraining(num_epochs=1, epoch_length=250, evaluate=False)
                sim_num += 1
                # with open('step_losses.pkl', 'wb') as f:
                #     pickle.dump(step_losses, f)
                
    def create_pretraining_datasets(
            self,
            random_settings,
            only_non_zero,
            num_sequential_sim_runs=5,
            label_date=None,
            next_step_offset=10
        ):
        
        i_episode = 0
        episode_start_balance = {'EUR': 1000, 'XRP': 0}
        
        while True:
                        
            sim_indexes = range(len(self.stream_dates))
            
            if label_date:
                sim_indexes.remove(self.stream_dates.index(label_date))
    
            for sim_index in sim_indexes:
                
                sim_numeric_data, sim_image_data, buy_rewards, sim_prices, weightedAvgPrice = \
                    prep_sim_data(sim_index, self.stream_dates, self.stream_times, self.data_path)
                            
                max_step = sim_numeric_data.shape[0]-next_step_offset
                
                for seq_index in range(num_sequential_sim_runs):

                    step_dict = {
                        'last_buy_atempt_step': 0,
                        'last_sell_attempt_step': 0,
                        'current_step': 0,
                        'base_balance_at_last_buy': self.balance['EUR']
                    }
                    
                    self.balance = copy.deepcopy(episode_start_balance)
                    base_balance_at_last_buy = self.balance['EUR']
    
                    last_buy_step, skip_next = 0, False
                    
                    r = random.sample([1000, 3000, 3000, 6000], 1)[0] # 3000 if i_episode % 3 == 0 else 6000
                    random_settings.r = r

                    print('Sim={}, seq_index={}, R={}, Episode {}'.format(
                        sim_index, seq_index, r, i_episode
                    ))

                    for current_step in range(max_step):
                        # add transaction delay
                        if skip_next:
                            skip_next = False
                        else:
                            state_tuple, sim_finished, reward, performed_action = self.perf_sim_step(
                                current_step,
                                sim_numeric_data,
                                sim_image_data,
                                sim_prices,
                                buy_rewards,
                                step_dict,
                                random_settings,
                                epsilon=1,
                                next_step_offset=next_step_offset
                            )                            

                            if ((not only_non_zero and random.random() <= 0.001) or
                                (only_non_zero and performed_action != 2)):
                                    
                                # add on avg every 1000th entry
                                self.memory.push(
                                    state_tuple,
                                    saving=True,
                                    save_prefix=self.play_path
                                )
                                print(
                                    f"\t\t Reward: {reward}, MemSize: {len(self.memory.memory)}, "
                                    f"Performed Action: {['buy', 'sell', 'do nothing'][performed_action]}"
                                )
                                
                            if current_step % 10000 == 0:
                                print(
                                    '\t\t Step={}, Balance={}'.format(
                                        current_step, self.balance
                                    )
                                ) 
     

                            if sim_finished:
                                break


                    print(
                        '\t\t Finishing sim {}. Start balance={},\n\t\t finishing balance={}\n\t\t final_step={}, max_step={}'.format(
                            sim_index, episode_start_balance, self.balance, current_step, sim_numeric_data.shape[0]
                        )
                    )
                    i_episode += 1
                
            
    def get_torch_state_batch(self, state_list, is_tensor=False):
        numeric_state_batch, image_state_batch = list(zip(*state_list))
        
        if not is_tensor:
            numeric_state_batch = torch.from_numpy(
                np.array(numeric_state_batch).astype(np.float16)
            ).to(self.device)
            
            image_state_batch = torch.from_numpy(
                np.array(image_state_batch).astype(np.float32)/255
            ).to(self.device)
            
        else:
            numeric_state_batch = torch.cat(numeric_state_batch)
            image_state_batch = torch.cat(image_state_batch)

        return numeric_state_batch, image_state_batch
    
    
    def pretraining(
        self,
        num_epochs=500,
        epoch_length=500,
        evaluate=False,
        include_zero=True
    ):
        datasets = os.listdir(self.play_path)
        non_zero_datasets = [d for d in datasets if 'non_zero' in d  and 'large' not in d]
        zero_datasets = [d for d in datasets if 'non_zero' not in d]
        
        for combo in [[self.optimizer, self.activation_function]]:#combinations: #zip(*[iter(combinations)]*3):
            
            #print('Combos: ', combo)
            print(str(combo[0]).split()[0], str(combo[0]).split("lr")[1].split()[1], str(combo[1]).split("function")[1].split()[0].strip('>'))
            
            last_epoch_loss = None
            epoch_losses = []

            for epoch in range(num_epochs):
                total_loss = 0
            
                for i in range(len(zero_datasets)):
                    train = []
                    test = []            
                    
                    nz = pickle.load(open(self.play_path+non_zero_datasets[i], 'rb'))
                    train.extend(nz)#[:4500])
                    
                    if include_zero:
                        z = pickle.load(open(self.play_path+zero_datasets[i], 'rb'))
                        train.extend(z)#[:4500])
                    
                    # test.extend(nz[4500:])
                    # test.extend(z[4500:])
                    
                   # self.memory.memory = train
                                
                    for j in range(epoch_length):
                        total_loss += float(self.optimize_model(
                            self.policy_net,
                            combo[0],
                            combo[1],
                            memory=train
                        ))#, j % 500 == 0))
                                                
                print('Pretraining Epoch={}, total loss={}'.format(
                    epoch, total_loss
                ))
                last_epoch_loss = round(total_loss, 4)
                epoch_losses.append(str(last_epoch_loss))                           
                
                if evaluate and epoch % 5 == 0 and epoch != 0:
                    self.training_loop(num_episodes=1, sim_indexes=None, no_random=True, optimize=False)       
                    
                    
            # df = df.append({
            #     'optimizer': str(combo[0]).split()[0],
            #     'learning rate': str(combo[0]).split("lr")[1].split()[1],
            #     'activation': str(combo[1]).split("function")[1].split()[0].strip('>'),
            #     'last_epoch_loss':last_epoch_loss,
            #     'epoch_losses': ', '.join(epoch_losses) 
            # }, ignore_index=True)
            # nets[0].apply(weight_reset)
            # df.to_csv('test_model_results_scale{}_noBN_tt.csv'.format(self.scale), index=False)
            
    def optimize_model(
            self,
            single_net=None,
            optimizer=None,
            activation_function=None,
            print_loss=False,
            memory=None
        ):
        
        if not memory:

            if len(self.memory.memory) < 10000: #self.batch_size:
                return
            
            transitions = random.sample(self.memory.memory, self.batch_size)
            transitions = transitions[:self.batch_size]
            
        else:
            transitions = random.sample(memory, self.batch_size)
        
        policy_net = single_net if single_net else self.policy_net
        target_net = single_net if single_net else self.target_net
        optimizer = optimizer if optimizer else self.optimizer
        activation_function = \
            activation_function if activation_function else self.activation_function
        
        batch = Transition(*zip(*transitions))        
        
        numeric_state_batch, image_state_batch = \
            self.get_torch_state_batch(batch.state)
            
        numeric_next_state_batch, image_next_state_batch = \
            self.get_torch_state_batch(batch.next_state)
            
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #Q(s_t, a) for each selected action 
        state_action_q_values = policy_net.forward(
            numeric_state_batch, image_state_batch, activation_function
        )[action_batch.max(1, keepdim=True)[0] == action_batch]

        next_state_action_q_values = target_net.forward(
            numeric_next_state_batch, image_next_state_batch, activation_function, disp=False
        ).max(1)[0].detach()

        # Compute the expected Q values = reward + discounted future rewards
        expected_state_action_values = reward_batch + (next_state_action_q_values * self.gamma)
        
        try:
            # Compute Huber loss
            loss = self.loss_function(
                state_action_q_values,
                expected_state_action_values.type(torch.cuda.FloatTensor)
            ).type(torch.cuda.FloatTensor)            
        
            if print_loss:
                print('Label Action Values: ', state_action_q_values)
                print('Pred Action Values: ', expected_state_action_values)
                print(loss)
                
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()  
            
            for var in [image_state_batch, image_next_state_batch]:
                var = var.cpu().detach().numpy() 
                del var           
            return loss
        
        except RuntimeError as e:
            print(e)
            print(
                'Input A (state_action_q_values) shape {},'
                'Input B (expected_state_action_values) shape {}'
                'Input B_0 (reward) shape {}'
                'Input B_1 (next_state_action_q_values) shape {}'.format(
                    state_action_q_values.shape,
                    expected_state_action_values.shape,
                    reward_batch.shape,
                    next_state_action_q_values.shape
            ))
            return 0

    def get_action(self,
                   state,
                   epsilon=1, # do random weights
                   selective_random=True,
                   r=6000):
        
        sample = random.random()
        
        if sample > epsilon:   
            numeric_input = torch.from_numpy(
                state[0]
            ).to(self.device).unsqueeze(dim=0)
            image_array = torch.from_numpy(
                state[1].astype(np.float16)/255
            ).to(self.device).unsqueeze(dim=0)

            
            with torch.no_grad():
                
                actions, cur_weights, fc_input = self.policy_net.forward(
                    numeric_input, image_array, self.activation_function, disp=False, return_extras=True
                )
                
                del numeric_input
                del image_array
                
                return actions, True, cur_weights, fc_input
            
        else:
            if selective_random:
                random_q = np.array([
                    np.random.uniform(0, 2, 1)[0],
                    np.random.uniform(0, 2, 1)[0],
                    np.random.uniform(0, r, 1)[0] #3000 avg = 30 minutes, 1500 avg 15 minutes huge variablility
                ])
            else:
                random_q = np.random.normal(
                    loc=1, scale=1.0, size=self.num_actions
                )
            
            return torch.tensor(
                random_q,
                device=self.device,
                dtype=torch.float
            ).unsqueeze(dim=0), False, None, None
        

    def get_large_reward_pretrain_data(self, large_limit=10):
        datasets = os.listdir(self.play_path)
        non_zero_datasets = [d for d in datasets if 'non_zero' in d ]
        
        large_reward_states = []
       
        for dataset in non_zero_datasets:
            states = pickle.load(open(self.play_path+dataset, 'rb'))
            
            rewards = Transition(*zip(*states))
            rewards = torch.abs(torch.cat(rewards.reward))
            
            reward_mask = rewards > large_limit

            print(rewards[reward_mask][:100])  
            
            selected_states = list(np.array(states)[reward_mask.cpu().numpy()])
            large_reward_states.extend(selected_states)
            
            print(len(selected_states))
        
        with open(self.play_path+'/non_zero_replay_large_rewards.pkl', 'wb') as f:
            pickle.dump(large_reward_states, f)
        
            
    class ReplayMemory():
        
        def __init__(self, capacity, play_path):
            self.memory = []
            self.capacity = capacity
            self.full = False
            self.save_count = 0
            self.stored_ratio = [0, 0, 0]  # buy, sell, do nothing
            self.states_processed = 0
            self.play_path = play_path
            
        def push(self, state_tuple, saving=False, save_prefix=''): #, readding_large_states=False):
            
            if self.full:
                self.memory.pop()
                
            self.memory.insert(0, state_tuple)
            # else:
            #     self.memory.append(state_tuple)
                
            if len(self.memory) >= self.capacity:
                if saving:
                    with open(save_prefix+'replay_{}.pkl'.format(self.save_count), 'wb') as f:
                        pickle.dump(self.memory, f)
                        
                    self.save_count += 1
                    self.memory = []
                else:
                    self.full = True
                    
            self.stored_ratio[torch.max(state_tuple[1], dim=1)[1]] += 1
            
            if self.states_processed % self.capacity == 0 and not saving:
                print('Adding large Reward States to memory..')
                self.add_large_reward_states_to_memory()

            self.states_processed += 1
                
        def sample(self, size):
            return random.sample(self.memory, size)
        
        def add_large_reward_states_to_memory(self, filename='non_zero_replay_large_rewards.pkl'):
            states = pickle.load(open(
                self.play_path+filename,
                'rb'
            ))
            
            # add to start of memory & pop last n states
            print('Size before =', len(self.memory))
            
            self.memory = states + self.memory
            
            if len(self.memory) > self.capacity:
                self.memory = self.memory[:self.capacity]  # pop extra states
                
            print('Size after =', len(self.memory))
        
        
    class DelayedStateAdder():
        
        def __init__(self,
                     steps_before_no_action_add=20):
            """
            Delays & stops no actions states being added to replay memory.
            Done as the ratio of no action to action can be huge 80000:1.
            
            Note:
                if steps_before_no_action_add = 10, only 1 in 10 no action steps 
                    wil be added to memory.

            Parameters
            ----------
            steps_before_no_action_add : int
                num steps before adding no action state to replay buffer

            """
            
            self.steps_before_no_action_add = steps_before_no_action_add
            self.delayed_states = []
            self.avg_steps_before_buy_sell = None
            self.last_buy_sell_step = 0
            
        def add_or_delay_state(self, step_num, state_tuple, memory):
            
            sel_action = torch.max(state_tuple[1], dim=1)[1]
            
            if sel_action == 2:
                #print('YESS')
                
                # if the ratio is not skewed just add as normal
                if self.avg_steps_before_buy_sell and \
                    self.avg_steps_before_buy_sell < self.steps_before_no_action_add:
                    memory.push(state_tuple)
                    
                else:
                    
                    self.delayed_states.append(state_tuple)
                    
                    if len(self.delayed_states) >= self.steps_before_no_action_add:
                        memory.push(random.sample(self.delayed_states, 1)[0])
                        self.delayed_states = []
                    
            else:
                memory.push(state_tuple)
                steps_before_buy_sell = step_num - self.last_buy_sell_step
                
                self.avg_steps_before_buy_sell = \
                    (steps_before_buy_sell*0.2 + self.avg_steps_before_buy_sell*0.8) \
                        if self.avg_steps_before_buy_sell else steps_before_buy_sell
                    
                self.last_buy_sell_step = step_num   
            
        
class RandomnessSettings():
    
    def __init__(self,
                 random_every_nth_sim=None,
                 no_epsilon_after_n_sims=40,
                 start_epsilon=1,
                 constant_epsilon=None,  # ignores other settings & sets constant epsilon
                 r=3000):
        
        self.random_every_nth_sim = random_every_nth_sim
        self.no_epsilon_after_n_sims = no_epsilon_after_n_sims
        self.start_epsilon = start_epsilon
        self.constant_epsilon = constant_epsilon
        self.r = r
        
        
    def get_epsilon(self, sim_num):
        
        epsilon =  (
            max((self.start_epsilon - 
                 sim_num/self.no_epsilon_after_n_sims), 0)
            
        ) if not self.constant_epsilon else self.constant_epsilon
        
        if self.random_every_nth_sim and \
            sim_num % self.random_every_nth_sim != 0:
                
            epsilon = 0 # only allow random every nth epoch
            
        return epsilon