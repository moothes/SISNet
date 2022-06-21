#! /bin/bash

for i in fca fbrs iog itsd minet ctdnet
do python3 test_fps.py $i --gpus=$1
done