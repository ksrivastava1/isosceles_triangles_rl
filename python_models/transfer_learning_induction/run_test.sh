#!/bin/bash

# untar your Python installation. Make sure you are using the right version!
tar -xzf python37.tar.gz
# (optional) if you have a set of packages (created in Part 1), untar them also
tar -xzf isosceles_triangles_env.tar.gz

# make sure the script will use your Python installation, 
# and the working directory as its home location
export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

# run your script
python3 run_test.py