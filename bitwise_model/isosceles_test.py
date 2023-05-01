import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import pickle
from numba import njit
import os

n_actions = 2 # number of actions that the agent can take. In this case, it is either 0 for excluding a point and 1 for including it
n_sessions = 2000 # number of new sessions per iteration
Learning_rate = 0.01 # learning rate, increase this to converge faster
percentile = 90 # top 100-x percentile the agent will learn from
super_percentile = 90 # top 100-x percentile of that survives to the next generation
Lambda = 0.8 # Weight for regularizing the reward function to generate more ones (too high a labda will result in higher odds of generating isosceles triangles)



@njit
def create_coordinates(n):
    coordinates = np.empty((n**2, 2))
    for i in range(n):
        for j in range(n):
            coordinates[i*n + j, 0] = i
            coordinates[i*n + j, 1] = j
    return coordinates


@njit
def euclidean_distance_matrix(n):
    # Create a meshgrid of the points
    coordinates = np.zeros((n * n, 2), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            coordinates[i * n + j, 0] = i
            coordinates[i * n + j, 1] = j

    # Compute pairwise Euclidean distances between points
    dist_matrix = np.zeros((n * n, n * n), dtype=np.float32)
    for i in range(n * n):
        for j in range(n * n):
            dx = coordinates[i, 0] - coordinates[j, 0]
            dy = coordinates[i, 1] - coordinates[j, 1]
            dist_matrix[i, j] = np.sqrt(dx * dx + dy * dy)

    return dist_matrix

# Helper function for counting isosceles triangles later

@njit
def count_elements(row):
    # Initialize an empty dictionary
    d = {}

    # Loop over each element in the row
    for x in row:
        # If the element is already in the dictionary, increment its count
        if x in d:
            d[x] += 1
        # Otherwise, add it to the dictionary with a count of 1
        else:
            d[x] = 1

    # Return the resulting dictionary
    return d

@njit
def count_isosceles_triangles(board, distance_matrix, p):

    # Find indices of all ones on the board
    one_indices = np.flatnonzero(board == 1)
    
    # Compute pairwise distances between all ones and store in a matrix
    # So row i is the distances from the ith one to all other ones

    ones_distance = np.zeros((p, p), dtype=np.float32)

    for i in range(p):
        for j in range(i+1,p):
            ones_distance[i,j] = distance_matrix[one_indices[i], one_indices[j]]
            ones_distance[j,i] = ones_distance[i,j]

    # Initialize count of isosceles triangles to zero
    count = 0

    # Check how many repeated distances there are in each row
    # Each repeated distance is an isoceles triangle
    for row in ones_distance:
        hash_table = count_elements(row)
        
        for key in hash_table:
            count += hash_table[key] - 1

    return count

@njit
def convert_to_board(word, n):
    word = word.astype(np.float32)  # cast input array to float32
    board = np.zeros((n, n), dtype=np.float32)
    for i in range(len(word)):
        board[i//n, i%n] = word[i]
    return board


@njit
def state_to_word(state, len_word):
    word = np.zeros(len_word, dtype=np.float32)
    for i in range(len_word):
        word[i] = state[i]
    for i in range(len_word, 2*len_word):
        if state[i] == 1:
            word[i - len_word] = 1
    return word

def convert_to_word(board, n):
    word = torch.zeros(n**2)
    for i in range(n):
        for j in range(n):
            word[i*n + j] = board[i, j]
    return word

@njit
def get_score(word, dist_matrix,len_word, n):

    word = word.astype(np.float32)  # cast input array to float32

    # If the size of the word is 2*len_word, then convert it to a word first
    if len(word) == 2*len_word:
        word = state_to_word(word, len_word)


    board = convert_to_board(word, n)

    # Get the number of 1s on the board
    num_ones = np.sum(board)

    # Calculate the number of isosceles triangles on the board
    isosceles = count_isosceles_triangles(board, dist_matrix, np.int(num_ones))

    return -1*(isosceles) + Lambda*(num_ones)

# Define a function to generate a new session

def generate_session(agent, n_sessions, n, dist_matrix, len_word, len_game, observation_space, verbose = 1):

    states = torch.zeros((n_sessions, observation_space, len_game), dtype=torch.float)
    actions = np.zeros([n_sessions, len_game], dtype = int)
    state_next = torch.zeros((n_sessions, observation_space), dtype=torch.float)
    prob = torch.zeros([n_sessions,1], dtype = torch.float)

    states[:,len_word,0] = 1

    step = 0
    total_score = np.zeros([n_sessions])
    recordsess_time = 0
    play_time = 0
    scorecalc_time = 0
    pred_time = 0

    terminal = False 

    while not terminal:

        step += 1		
        tic = time.time()
        prob = agent(states[:,:,step-1])
        pred_time += time.time()-tic

        for i in range(n_sessions):
            
            if np.random.rand() < prob[i]:
                action = 1
            else:
                action = 0      
            
            actions[i][step-1] = action

            tic = time.time()
            state_next[i] = states[i,:,step-1]
            play_time += time.time()-tic

            if (action > 0):
                state_next[i][step-1] = action		         
            state_next[i][len_word + step-1] = 0

            if (step < len_word):
                state_next[i][len_word + step] = 1			
            terminal = step == len_word

            if terminal:
                total_score[i] = get_score(state_next[i].numpy(), dist_matrix, len_word, n)
            scorecalc_time += time.time()-tic
            tic = time.time()

            if not terminal:
                states[i,:,step] = state_next[i]			
            recordsess_time += time.time()-tic
        
        if terminal:
            break
	#If you want, print out how much time each step has taken. This is useful to find the bottleneck in the program.	
    if verbose == 1:
        print("Predict: "+str(pred_time)+", play: " + str(play_time) +", scorecalc: " + str(scorecalc_time) +", recordsess: " + str(recordsess_time))
    return states, actions, total_score

# Define a function 

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


def train(board_size, write_all, write_best, filename):

    n = board_size
    dist_matrix = euclidean_distance_matrix(n)
    len_word = np.power(n, 2) # length of the word that we want to generate. The word we generate is of length n^2 with a 1 for whether it is in the subset and 0 if not
    observation_space = 2*len_word #Leave this at 2*len_word. The input vector will have size 2*len_word, where the first MYN letters encode our partial word (with zeros on
                            #the positions we haven't considered yet), and the next MYN bits one-hot encode which letter we are considering now.
                            #So e.g. [0,1,0,0,   0,0,1,0] means we have the partial word 01 and we are considering the third letter now.
                            #Is there a better way to format the input to make it easier for the neural network to understand things?
    len_game = len_word 

    INF = 1000000

    first_layer_neurons = 128
    second_layer_neurons = 64
    third_layer_neurons = 4


    #Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
    #I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
    #It is important that the loss is binary cross-entropy if alphabet size is 2.

    # Define the neural network architecture
    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.fc1 = nn.Linear(observation_space, first_layer_neurons)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(first_layer_neurons, second_layer_neurons)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(second_layer_neurons, third_layer_neurons)
            self.relu = nn.ReLU()
            self.fc4 = nn.Linear(third_layer_neurons, 1)
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

    global super_states
    super_states = torch.empty((0, len_game, observation_space), dtype=torch.int)
    global super_actions
    super_actions = torch.tensor([], dtype=torch.int)
    global super_rewards
    super_rewards = torch.tensor([])
    sessgen_time = 0
    fit_time = 0
    score_time = 0

    pass_threshold =  1.5 # the mean_all_reward must be greater than this threshold to pass the test

    counter = 0 # Counter for number of generations where the reward is < 1. 

    for i in range(20000):
        tic = time.time()
        sessions = generate_session(net,n_sessions,n, dist_matrix, len_word, len_game, observation_space, 1) #change 0 to 1 to print out how much time each step in generate_session takes 
        sessgen_time = time.time()-tic

        tic = time.time()

        states_batch = torch.tensor(sessions[0], dtype=torch.int)
        actions_batch = torch.tensor(sessions[1], dtype=torch.int)
        rewards_batch = torch.tensor(sessions[2])
        states_batch = torch.transpose(states_batch, 1, 2)

        states_batch = torch.cat((states_batch, super_states), dim=0)

        

        if i>0:
            actions_batch = torch.cat((actions_batch, super_actions), dim=0)

        rewards_batch = torch.cat((rewards_batch, super_rewards), dim=0)

        randomcomp_time = time.time()-tic 
        tic = time.time()

        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
        select1_time = time.time()-tic

        tic = time.time()
        super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
        select2_time = time.time()-tic

        tic = time.time()
        super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
        super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
        select3_time = time.time()-tic

        tic = time.time()
        optimizer.zero_grad()
        # Forward pass
        outputs = net(elite_states)
        # Calculate the loss
        loss = criterion(outputs, elite_actions.unsqueeze(1))
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        fit_time = time.time()-tic

        tic = time.time()

        super_states = torch.stack([super_sessions[i][0] for i in range(len(super_sessions))])
        super_actions = torch.stack([super_sessions[i][1] for i in range(len(super_sessions))])
        super_rewards = torch.stack([super_sessions[i][2] for i in range(len(super_sessions))])


        rewards_batch.sort()
        mean_all_reward = torch.mean(rewards_batch[-100:])	
        mean_best_reward = torch.mean(super_rewards)	

        if mean_all_reward < 1:
            counter += 1

        score_time = time.time()-tic
        
        print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))

        #uncomment below line to print out how much time each step in this loop takes. 
        print(	"Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 

        # Make a new folder if 'Data' folder does not exist
        if not os.path.exists('Data'):
            os.makedirs('Data')

        if write_all:
            if (i%20 == 1): #Write all important info to files every 20 iterations
                with open(os.path.join('Data', str(filename)+'_best_species_pickle'+'.txt'), 'wb') as fp:
                    pickle.dump(super_actions, fp)
                with open(os.path.join('Data', str(filename)+'_best_species'+'.txt'), 'w') as f:
                    for item in super_actions:
                        f.write(str(convert_to_board(item.numpy(), n)))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_species_rewards'+'.txt'), 'w') as f:
                    for item in super_rewards:
                        f.write(str(item))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_100_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_all_reward)+"\n")
                with open(os.path.join('Data', str(filename)+'_best_elite_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_best_reward)+"\n")
                if (i%200==2): # To create a timeline, like in Figure 3
                    with open(os.path.join('Data', str(filename)+'_best_species_timeline'+'.txt'), 'a') as f:
                        f.write(str(convert_to_board(super_actions[0].numpy(), n)))
                        f.write("\n")
        if write_best:
            if mean_all_reward > pass_threshold:
                with open(os.path.join('Data', str(filename)+'_best_species_pickle'+'.txt'), 'wb') as fp:
                    pickle.dump(super_actions, fp)
                with open(os.path.join('Data', str(filename)+'_best_species'+'.txt'), 'w') as f:
                    for item in super_actions:
                        f.write(str(convert_to_board(item.numpy(), n)))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_species_rewards'+'.txt'), 'w') as f:
                    for item in super_rewards:
                        f.write(str(item))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_100_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_all_reward)+"\n")
                with open(os.path.join('Data', str(filename)+'_best_elite_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_best_reward)+"\n")
                with open(os.path.join('Data', str(filename)+'_best_species_timeline'+'.txt'), 'a') as f:
                    f.write(str(convert_to_board(super_actions[0].numpy(), n)))
                    f.write("\n")
        
        if counter > 100 and mean_all_reward > pass_threshold:
            with open(os.path.join('Data', str(filename)+'_best_species_pickle'+'.txt'), 'wb') as fp:
                pickle.dump(super_actions, fp)
            with open(os.path.join('Data', str(filename)+'_best_species'+'.txt'), 'w') as f:
                for item in super_actions:
                    f.write(str(convert_to_board(item.numpy(), n)))
                    f.write("\n")
            with open(os.path.join('Data', str(filename)+'_best_species_rewards.txt'), 'w') as f:
                for item in super_rewards:
                    f.write(str(item))
                    f.write("\n")
            with open(os.path.join('Data', str(filename)+'_best_100_rewards'+'.txt'), 'a') as f:
                f.write(str(mean_all_reward)+"\n")
            with open(os.path.join('Data', str(filename)+'_best_elite_rewards'+'.txt'), 'a') as f:
                f.write(str(mean_best_reward)+"\n")
            with open(os.path.join('Data', str(filename)+'_best_species_timeline'+'.txt'), 'a') as f:
                f.write(str(convert_to_board(super_actions[0].numpy(), n)))
                f.write("\n")
            return super_actions[0].numpy()
        
                    
