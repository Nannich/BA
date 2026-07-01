#!/bin/bash
set -e

DATASET="mCAD-2000-1"
TEST_GENE="1"

echo -e "\n[1/4] Testing Preprocessing"
python3 main.py process $DATASET


echo -e "\n[2/4] Testing Trajectory"
python3 main.py trajectory train $DATASET --gene all --model kan --loss mse
python3 main.py trajectory train $DATASET --gene $TEST_GENE --model mlp --loss zinb
python3 main.py trajectory train $DATASET --gene $TEST_GENE --model null --loss mse

python3 main.py trajectory eval $DATASET

python3 main.py trajectory plot $DATASET --mode scatter --gene $TEST_GENE
python3 main.py trajectory plot $DATASET --mode distribution --gene $TEST_GENE
python3 main.py trajectory plot $DATASET --mode trajectory --gene $TEST_GENE --model kan --loss mse
python3 main.py trajectory plot $DATASET --mode cluster --model kan --loss mse


echo -e "\n[3/4] Testing GRN"
python3 main.py grn extract $DATASET --input_mode smooth --target_mode log --loss mse --lag 0.0
python3 main.py grn extract $DATASET --input_mode log --target_mode log --loss zinb --lag 0.1 --ground_truth

python3 main.py grn plot $DATASET --arch smo_log --sensitivity 0.10
python3 main.py grn plot $DATASET --arch log_log_l_ground_truth --sensitivity 0.15
python3 main.py grn plot $DATASET --ground_truth


echo -e "\n[4/4] Testing Symbolic"
python3 main.py symbolic extract $DATASET --mode grn --arch smo_log
python3 main.py symbolic plot $DATASET --checkpoint models/$DATASET/symbolic/smo_log/deep_Pax6_checkpoint.pth