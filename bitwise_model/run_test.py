from isosceles_test import *

def main():

    n_low = 5 # board size lower bound (pick a board size above 3)
    n_up = 9 # board size upper bound (pick a board size above 3)
    write_all = False
    write_best = True

    for i in range(n_low,n_up):
        filename = str(i) + "x" + str(i)
        train(i, write_all, write_best, filename)

if __name__ == "__main__":
    main()

