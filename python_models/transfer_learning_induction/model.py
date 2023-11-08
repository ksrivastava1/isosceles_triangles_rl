import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from reward_function import *

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

n_sessions = 2000 # number of new sessions per iteration
Learning_rate = 0.001 # learning rate, increase this to converge faster
percentile = 99 # top 100-x percentile the agent will learn from
super_percentile = 90 # top 100-x percentile of that survives to the next generation

copy_time = 0
add_time = 0
count_time = 0

# Defining a helper function that takes in a game and outputs the final board state
def final_board_state(game):
    n = len(game)
    for i in range(len(game)):
        if i == 0:
            continue
        if i == n-1:
            return game[i]
        if game[i+1].sum() == 0 and game[i].sum() != 0:
            return game[i]

# Defining a helper function to add points on the board based on a probability output from the neural network
def add_points(input_state, action_vec):

    global copy_time
    global add_time
    global count_time

    tic = time.time()
    n = int(np.sqrt(len(input_state)))

    point_added = False
    action_taken = np.zeros([len(action_vec)])
    cur_state = torch.clone(input_state)
    ones_indices = torch.nonzero(cur_state).numpy()
    copy_time += time.time() - tic

    tic = time.time()
    while not point_added:

        action_index = torch.multinomial(action_vec, 1).item()

        if cur_state[action_index] == 0:
            cur_state[action_index] = 1
            action_taken[action_index] = 1
            point_added = True
        else:
            action_vec[action_index] = 0
            action_vec = action_vec / torch.sum(action_vec)

    terminal = False
    add_time += time.time() - tic

    tic = time.time()
    if point_added:
        terminal = count_isosceles_triangles(ones_indices, action_index, n) > 0

    count_time += time.time() - tic
    return cur_state, action_taken, terminal

# Defining a function to generate a new session
def generate_session(agent, n_sessions, n, verbose = 1):
    global copy_time
    global add_time
    global count_time

    states = torch.zeros((n_sessions, n*n, n*n), dtype=torch.float)
    states.to(device)

    actions = np.zeros([n_sessions, n*n, n*n])

    total_score = np.zeros([n_sessions])
    recordsess_time = 0
    scoreUpdate_time = 0
    play_time = 0
    pred_time = 0

    copy_time = add_time = count_time = 0
    for i in range(n_sessions):

        terminal = False
        step = 0

        while not terminal:
            step+=1
            cur_state = states[i,step-1, :]

            tic = time.time()
            output = agent(cur_state)
            pred_time += time.time() - tic

            tic = time.time()
            new_state, action, terminal = add_points(cur_state, output)
            play_time += time.time() - tic

            tic = time.time()
            if terminal:
                total_score[i] = cur_state.sum()
                continue
            actions[i,step-1, :] = action
            scoreUpdate_time += time.time() - tic

            tic = time.time()
            states[i,step, :] = new_state
            recordsess_time += time.time() - tic

    print("Copy: " + str(copy_time) + " Add: " + str(add_time) + " Count: " + str(count_time))

    if verbose == 1:
        print("Predict: " + str(pred_time) + " Play: " + str(play_time) + " ScoreUpdate: " + str(scoreUpdate_time) + " Record: " + str(recordsess_time))

    return states, torch.tensor(actions, dtype = torch.int), torch.tensor(total_score)

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):

    counter = n_sessions * (100 - percentile)/100
    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_states = torch.empty(0)
    elite_actions = torch.empty(0)

    for i in range(len(states_batch)):

        if counter <= 0:
            break

        if rewards_batch[i] >= reward_threshold - 0.01:
            game_end_index = 0
            for item in states_batch[i]:
                if item.sum() == 0 and game_end_index != 0:
                    break
                elite_states = torch.cat((elite_states, item.unsqueeze(0)))
                game_end_index += 1

            for item in actions_batch[i]:
                if game_end_index == 0:
                    break
                elite_actions = torch.cat((elite_actions, item.unsqueeze(0)))
                game_end_index -= 1
            counter -= 1

    return elite_states, elite_actions


def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):

    counter = n_sessions * (100 - percentile)/100
    reward_threshold = np.percentile(rewards_batch, percentile)

    super_states = torch.empty(0)
    super_actions = torch.empty(0)
    super_rewards = torch.empty(0)

    for i in range(len(states_batch)):

        if counter <= 0:
            break

        if rewards_batch[i] >= reward_threshold - 0.001:
            super_states = torch.cat((super_states, states_batch[i].unsqueeze(0)), dim=0)
            super_actions = torch.cat((super_actions, actions_batch[i].unsqueeze(0)), dim=0)
            super_rewards = torch.cat((super_rewards, torch.tensor([rewards_batch[i]])), dim=0)
            counter -= 1


    return super_states, super_actions, super_rewards




def train(board_size, write_all, write_best, filename, slow):

    n = board_size

    input_space = n*n

    INF = 1000000

    verbose = 1 #set to 1 to print out how much time each step in generate_session takes


    first_layer_neurons = 128
    second_layer_neurons = 64
    third_layer_neurons = 4
    last_layer_neurons = n*n



    #Model structure: a sequential network with three hidden layers, sigmoid activation in the output.
    #I usually used relu activation in the hidden layers but play around to see what activation function and what optimizer works best.
    #It is important that the loss is binary cross-entropy if alphabet size is 2.

    # Defining the neural network architecture
    class MyNet(nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.fc1 = nn.Linear(input_space, first_layer_neurons)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(first_layer_neurons, second_layer_neurons)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(second_layer_neurons, third_layer_neurons)
            self.relu = nn.ReLU()
            self.fc4 = nn.Linear(third_layer_neurons, last_layer_neurons)
            self.sigmoid = nn.Softmax(dim=0)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.sigmoid(self.fc4(x))

            return x

    # Create an instance of the neural network
    net = MyNet()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=Learning_rate)

    global super_states
    super_states = torch.empty((0, n*n, n*n), dtype=torch.int)
    global super_actions
    super_actions = torch.tensor([], dtype=torch.int)
    global super_rewards
    super_rewards = torch.tensor([])
    sessgen_time = 0
    fit_time = 0
    score_time = 0

    counter = 0
    pass_threshold = 0.4*n

    cur_best_reward = 0
    cur_best_board = torch.zeros([n*n])
    cur_best_game = torch.zeros([n*n, n*n])
    cur_best_actions = torch.zeros([n*n, n*n])

    for i in range(10000):
        tic = time.time()
        states_batch, actions_batch, rewards_batch = generate_session(net, n_sessions, n, verbose)
        sessgen_time = time.time() - tic

        states_batch = states_batch.to(dtype=torch.int)

        tic = time.time()
        if i >0:
            states_batch = torch.cat((cur_best_game.reshape(1, n*n, n*n), states_batch, super_states), dim=0)
            actions_batch = torch.cat((cur_best_actions.reshape(1, n*n, n*n), actions_batch, super_actions), dim=0)
            rewards_batch = torch.cat((torch.tensor([cur_best_reward]), rewards_batch, super_rewards), dim=0)
        randomcomp_time = time.time() - tic

        tic = time.time()
        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile = percentile)
        select1_time = time.time() - tic

        tic = time.time()
        super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile = super_percentile)
        select2_time = time.time() - tic

        tic = time.time()
        super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
        super_sessions.sort(key=lambda x: x[2], reverse=True)
        select3_time = time.time() - tic

        tic = time.time()
        optimizer.zero_grad()

        elite_states.to(device)
        outputs = net(elite_states)

        loss = criterion(outputs, elite_actions.float())

        loss.backward()
        optimizer.step()
        fit_time += time.time() - tic

        tic = time.time()

        super_states = torch.stack([super_sessions[i][0] for i in range(len(super_sessions))])
        super_actions = torch.stack([super_sessions[i][1] for i in range(len(super_sessions))])
        super_rewards = torch.stack([super_sessions[i][2] for i in range(len(super_sessions))])

        mean_all_reward = torch.mean(rewards_batch[-100:])
        mean_best_reward = torch.mean(super_rewards)

        score_time == time.time() - tic

        if mean_best_reward > 1.25*n:
            counter+=1

        print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))

        #uncomment below line to print out how much time each step in this loop takes.
        print(	"Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time))

        #uncomment below line to print out the mean best reward
        print("Mean best reward: " + str(mean_best_reward))

        # Make a new folder if 'Data' folder does not exist
        if not os.path.exists('Data'):
            os.makedirs('Data')

        max_index = torch.argmax(super_rewards)

        if super_rewards[max_index] > cur_best_reward:
            cur_best_reward = super_rewards[max_index]
            cur_best_board = final_board_state(super_states[max_index]).numpy()
            cur_best_game = super_states[max_index]
            cur_best_actions = super_actions[max_index]

            with open(os.path.join('Data', str(filename)+'_best_board_timeline'+'.txt'), 'a') as f:
                f.write(str(convert_to_board(cur_best_board, n)))
                f.write("\n")

        if write_all:
            if (i%20 == 1): #Write all important info to files every 20 iterations
                with open(os.path.join('Data', str(filename)+'_best_species'+'.txt'), 'w') as f:
                    for game in super_states:
                        f.write(str(convert_to_board(final_board_state(game).numpy(), n)))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_species_rewards'+'.txt'), 'w') as f:
                    for item in super_rewards:
                        f.write(str(item))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_100_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_all_reward)+"\n")
                with open(os.path.join('Data', str(filename)+'_best_super_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_best_reward)+"\n")
                if (i%200==2):
                    with open(os.path.join('Data', str(filename)+'_best_species_timeline'+'.txt'), 'a') as f:
                        f.write(str(convert_to_board(final_board_state(super_states[max_index]).numpy(), n)))
                        f.write("\n")
        if write_best:
            if mean_best_reward > pass_threshold:
                with open(os.path.join('Data', str(filename)+'_best_species'+'.txt'), 'w') as f:
                    for game in super_states:
                        f.write(str(convert_to_board(final_board_state(game).numpy(), n)))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_species_rewards'+'.txt'), 'w') as f:
                    for item in super_rewards:
                        f.write(str(item))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_100_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_all_reward)+"\n")
                with open(os.path.join('Data', str(filename)+'_best_super_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_best_reward)+"\n")
                if (i%200==2):
                    max_index = torch.argmax(super_rewards)
                    with open(os.path.join('Data', str(filename)+'_best_species_timeline'+'.txt'), 'a') as f:
                        f.write(str(convert_to_board(final_board_state(super_states[max_index]).numpy(), n)))
                        f.write("\n")

        if counter > 1000:
            if mean_best_reward > pass_threshold:
                with open(os.path.join('Data', str(filename)+'_best_species'+'.txt'), 'w') as f:
                    for game in super_states:
                        f.write(str(convert_to_board(final_board_state(game).numpy(), n)))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_species_rewards'+'.txt'), 'w') as f:
                    for item in super_rewards:
                        f.write(str(item))
                        f.write("\n")
                with open(os.path.join('Data', str(filename)+'_best_100_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_all_reward)+"\n")
                with open(os.path.join('Data', str(filename)+'_best_super_rewards'+'.txt'), 'a') as f:
                    f.write(str(mean_best_reward)+"\n")
                if (i%200==2):
                    max_index = torch.argmax(super_rewards)
                    with open(os.path.join('Data', str(filename)+'_best_species_timeline'+'.txt'), 'a') as f:
                        f.write(str(convert_to_board(final_board_state(super_states[max_index]).numpy(), n)))
                        f.write("\n")
            return net
