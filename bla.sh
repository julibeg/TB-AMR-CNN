#!/bin/bash

sed -E 's/WEIGHTED_LOSS = \w+$/WEIGHTED_LOSS = True/' -i code/train.py
python code/train.py
rename 's/$/_w/' saved_models/*

sed -E 's/WEIGHTED_LOSS = \w+$/WEIGHTED_LOSS = False/' -i code/train.py
python code/train.py
rename 's/$/_uw/' saved_models/*
rename 's/w_uw$/w/' saved_models/*
