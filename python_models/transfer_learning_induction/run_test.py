from model import *
import torch

def main():

    n_low = 50 # board size lower bound (pick a board size above 3)
    n_up = 51 # board size upper bound (pick a board size above 3)
    write_all = True
    write_best = True
    slow = False # This is for the manual counting method which is more space efficient but slower

    for i in range(n_low,n_up):
        filename = str(i) + "x" + str(i)
        train(i, write_all, write_best, filename, slow)
        


if __name__ == "__main__":
    main()
