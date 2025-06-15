#!/bin/bash

for i in {1..5}
do
    result=$(python3 train.py)
    python3 eval.py "$result"
done