import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
from numba import njit
import os
from reward_function import *

n_sessions = 2000 # number of new sessions per iteration
Learning_rate = 0.01 # learning rate, increase this to converge faster
percentile = 90 # top 100-x percentile the agent will learn from
super_percentile = 90 # top 100-x percentile of that survives to the next generation
Lambda = 0.8 # Weight for regularizing the reward function to generate more ones (too high a labda will result in higher odds of generating isosceles triangles)

# Defining a function to generate a new session

def generate_session(agent, n_sessions, n, dist_matrix, len_word, len_game, observation_space, verbose = 1, slow = True):

    boards = torch.zeros((n_sessions, observation_space, len_game), dtype=torch.float32)
    actions = np.zeros((n_sessions, len_game), dtype=np.int32)
    boards_next = torch.zeros((n_sessions, observation_space), dtype=torch.float32)
    prob = torch.zeros((n_sessions, 1), dtype=torch.float32)

    for i in range(20000):
        tic = time.time()
        

    
    return boards


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):

    counter = n_sessions * (100 - percentile)/100
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = torch.empty(0)
    elite_actions = torch.empty(0)

    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold - 0.000001 and counter > 0:
            for item in states_batch[i]:
                elite_states = torch.cat((elite_states, item.unsqueeze(0))) 
            for item in actions_batch[i]:
                elite_actions = torch.cat((elite_actions, item.unsqueeze(0))) 		
        counter -= 1
	
    return elite_states, elite_actions


def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):

    counter = n_sessions * (100 - percentile)/100
    reward_threshold = np.percentile(rewards_batch, percentile)

    super_states = torch.empty(0)
    super_actions = torch.empty(0)
    super_rewards = torch.empty(0)

    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold - 0.000001 and counter > 0:
            super_states = torch.cat((super_states, states_batch[i].unsqueeze(0)), dim=0)
            super_actions = torch.cat((super_actions, actions_batch[i].unsqueeze(0)), dim=0)
            super_rewards = torch.cat((super_rewards, rewards_batch[i].unsqueeze(0)), dim=0) 			
        counter -= 1
    

    return super_states, super_actions, super_rewards




def train(board_size, write_all, write_best, filename, slow):

    n = board_size
    if slow:
        dist_matrix = generate_distance_matrix(1)
    else:
        dist_matrix = generate_distance_matrix(n)
    len_word = np.power(n, 2) # length of the word that we want to generate. The word we generate is of length n^2 with a 1 for whether it is in the subset and 0 if not
    observation_space = 2*len_word #Leave this at 2*len_word. The input vector will have size 2*len_word, where the first MYN letters encode our partial word (with zeros on
                            #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
                            #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
                            #Is there a better way to format the input to make it easier for the neural network to understand things?
    len_game = len_word 

    INF = 1000000

    max_size = 5 # This indicates the largest size 

    first_layer_neurons = 1
    second_layer_neurons = 128
    third_layer_neurons = 64
    last_layer_neurons = 9



    #Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
    #I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
    #It is important that the loss is binary cross-entropy if alphabet size is 2.

    # Defining the neural network architecture
    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.fc1 = nn.Linear(observation_space, first_layer_neurons)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(first_layer_neurons, second_layer_neurons)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(second_layer_neurons, third_layer_neurons)
            self.relu = nn.ReLU()
            self.fc4 = nn.Linear(third_layer_neurons, last_layer_neurons)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.sigmoid(self.fc4(x))

            return x

    # Create an instance of the neural network
    net = MyNet()

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=Learning_rate)

    for i in range(10000):
        states



    





    

