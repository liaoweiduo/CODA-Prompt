#!/bin/bash
# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0'
CONFIG=configs/cifar-100_prompt.yaml
CONFIG_SLOT=configs/cifar-100_slot.yaml
CONFIG_FT=configs/cifar-100_ft.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR
# SLOT-Prompt
#
# prompt parameter args:
#    arg 1 = prompt component pool size, no use
#    arg 2 = prompt length
#    arg 3 = num of slots extracted from one img
#    arg 4 = coeff for regularization
#    arg 5 = p
#    --oracle_flag --upper_bound_flag \
#    --debug_mode 1 \
#for coeff in 0.001 0.003 0.005 0.007 0.009
#do
LOGNAME=slot-k10-nt5-recon-slr1e-4
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type slotmo --learner_name SLOTPrompt \
    --prompt_param 30 40 10 5 1.0 1.0 0.0 0.0 0.1 1.2 \
    --slot_lr 1e-4 \
    --only_learn_slot \
    --log_dir ${OUTDIR}/${LOGNAME}
date
#done

## CODA-P
##
## prompt parameter args:
##    arg 1 = prompt component pool size
##    arg 2 = prompt length
##    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name CODAPrompt \
#    --prompt_param 100 8 0.0 \
#    --log_dir ${OUTDIR}/coda-p
#
## DualPrompt
##
## prompt parameter args:
##    arg 1 = e-prompt pool size (# tasks)
##    arg 2 = e-prompt pool length
##    arg 3 = g-prompt pool length
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name DualPrompt \
#    --prompt_param 10 20 6 \
#    --log_dir ${OUTDIR}/dual-prompt
#
## L2P++
##
## prompt parameter args:
##    arg 1 = e-prompt pool size (# tasks)
##    arg 2 = e-prompt pool length
##    arg 3 = -1 -> shallow, 1 -> deep
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name L2P \
#    --prompt_param 30 20 -1 \
#    --log_dir ${OUTDIR}/l2p++
#
## FT
#python -u run.py --config $CONFIG_FT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type default --learner_name FinetunePlus \
#    --log_dir ${OUTDIR}/ft++
#
## FT++
#python -u run.py --config $CONFIG_FT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type default --learner_name NormalNN \
#    --log_dir ${OUTDIR}/ft
#
## Offline
#python -u run.py --config $CONFIG_FT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type default --learner_name NormalNN --upper_bound_flag \
#    --log_dir ${OUTDIR}/offline

