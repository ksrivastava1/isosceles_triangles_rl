from numba import njit
import numpy as np
from numba import prange
import torch
import torch.nn as nn
import time

board_construction = 'rowwise' # 'rowwise' or 'spiral'
board_type = 'euclidean' # 'euclidean' or 'torus'

# Helper function for computing torus distance for later

@njit
def torus_distance(i1, j1, i2, j2, rows, cols):
    # Calculate the minimum distance horizontally
    d1 = abs(j2 - j1)
    d1 = min(d1, cols - d1)

    # Calculate the minimum distance vertically
    d2 = abs(i2 - i1)
    d2 = min(d2, rows - d2)

    # Calculate the torus distance using Pythagorean theorem
    return np.sqrt(d1**2 + d2**2)

@njit
def convert_to_board(word, n):
    word = word.astype(np.float32)  # cast input array to float32
    board = np.zeros((n, n), dtype=np.float32)
    if board_construction == 'rowwise':
        for i in prange(len(word)):
            board[i//n, i%n] = word[i]
    elif board_construction == 'spiral':
        i,j = 0,0 # starting coordinates
        direction = 'right'
        visited = np.zeros((n,n), dtype=np.float32)
        for k in prange(len(word)):
            board[i,j] = word[k]
            visited[i,j] = 1
            if direction == 'right':
                if j + 1 < n and visited[i,j+1] == 0:
                    j += 1
                else:
                    direction = 'down'
                    i += 1
            elif direction == 'down':
                if i + 1 < n and visited[i+1,j] == 0:
                    i += 1
                else:
                    direction = 'left'
                    j -= 1
            elif direction == 'left':
                if j - 1 >= 0 and visited[i,j-1] == 0:
                    j -= 1
                else:
                    direction = 'up'
                    i -= 1
            elif direction == 'up':
                if i - 1 >= 0 and visited[i-1,j] == 0:
                    i -= 1
                else:
                    direction = 'right'
                    j += 1
    return board

@njit
def count_isosceles_triangles(one_indices, new_point_index, n):
    if board_type == 'euclidean':
        p = len(one_indices)

        # Initialize count of isosceles triangles to zero
        count = 0

        # Iterate over all pairs of ones
        for i in prange(p):
            for j in prange(i + 1, p):
                # Calculate the distance between each pair of ones
                dist_ij = np.sqrt((one_indices[i] // n - one_indices[j] // n) ** 2 +
                                (one_indices[i] % n - one_indices[j] % n) ** 2)
                dist_ik = np.sqrt((one_indices[i] // n - new_point_index // n ) ** 2 +
                                (one_indices[i] % n - new_point_index % n) ** 2)
                dist_jk = np.sqrt((one_indices[j] // n - new_point_index // n ) ** 2 +
                                (one_indices[j] % n - new_point_index % n) ** 2)

                # Check if any two distances are equal (up to a tolerance)
                if dist_ij == dist_ik or dist_ij == dist_jk or dist_ik == dist_jk:
                    count += 1
    elif board_type == 'torus':

        # Initialize count of isosceles triangles to zero
        count = 0

        # Iterate over all triples of ones
        for i in prange(p):
            for j in prange(i + 1, p):
                
                # Calculate the torus distance between each pair of ones
                dist_ij = torus_distance(one_indices[i] // n, one_indices[i] % n,
                                            one_indices[j] // n, one_indices[j] % n,
                                            n, n)
                dist_ik = torus_distance(one_indices[i] // n, one_indices[i] % n,
                                            new_point_index // n, new_point_index % n,
                                            n, n)
                dist_jk = torus_distance(one_indices[j] // n, one_indices[j] % n,
                                            new_point_index // n, new_point_index % n,
                                            n, n)

                # Check if any two distances are equal (up to a tolerance)
                if dist_ij == dist_ik or dist_ij == dist_jk or dist_ik == dist_jk:
                    count += 1

    return count

def copy_and_freeze_weights(net1, net2):
    # Verify that the networks have the same number of layers
    if len(list(net1.children())) != len(list(net2.children())):
        raise ValueError("The networks must have the same number of layers.")

    for layer1, layer2 in zip(net1.children(), net2.children()):
        # Check if the layers have the same input and output dimensions
        if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
            in_features1, out_features1 = layer1.in_features, layer1.out_features
            in_features2, out_features2 = layer2.in_features, layer2.out_features

            if in_features2 >= in_features1 and out_features2 >= out_features1:
                # Copy the weights and freeze them
                layer2.weight.data[:out_features1, :in_features1] = layer1.weight.data
                layer2.bias.data[:out_features1] = layer1.bias.data

                # Freeze the copied weights in net2
                layer2.weight.requires_grad = False
                layer2.bias.requires_grad = False
            else:
                raise ValueError("Layer dimensions in net2 should be wider than in net1.")

        else:
            raise ValueError("Both networks should consist of nn.Linear layers only.")
