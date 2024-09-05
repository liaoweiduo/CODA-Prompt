# bash experiments/imagenet-r.sh
# experiment settings
DATASET=CGQA
N_CLASS=100

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0'   # '0 1 2 3'
CONFIG_SLOT=configs/cgqa_slot.yaml
CONFIG=configs/cgqa_prompt.yaml
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
#    --oracle_flag --upper_bound_flag \
#    --debug_mode 1 \
LEARNERTYPE=slotmo
LEARNERNAME=SLOTPrompt
#for coeff in 0.02 0.04 0.06 0.08 0.1
for lr in 0.0001 0.0005 0.001 0.005
do
LOGNAME=slot-k5-recon-l2weight-coeff0.05-lr${lr}
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
    --prompt_param 100 8 5 0.05 \
    --lr ${lr} \
    --log_dir ${OUTDIR}/${LOGNAME}
date
done

# PMO-Prompt
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#    arg 4 = num of objectives: 2
#    arg 5 = mask: 0.0; -10000: randn; -10001: uniform; -10002: ortho; -10003: None
#    arg 6 = mask_mode: 0: maskout or 1: use
#    arg 7 = hv coeff, -1 to use LCQP
#    --oracle_flag --upper_bound_flag \
#LEARNERTYPE=pmo
#LEARNERNAME=PMOPrompt
#LOGNAME=pmo-f4m-epoch30-first30-min2
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
#    --prompt_param 21 2 0.0 2 -10003 1 -1 \
#    --log_dir ${OUTDIR}/${LOGNAME}
#date

# CODA-P-Replay
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#LEARNERNAME=CODAPromptR
#LOGNAME=coda-p-r-0-ortho
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 100 8 0.0 \
#    --memory 0 \
#    --log_dir ${OUTDIR}/${LOGNAME}
#date

#LEARNERNAME=CODAPrompt
#LOGNAME=coda-p-r-0-ortho
#for mode in sys pro sub non noc
#do
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name ${LEARNERNAME} \
#      --prompt_param 100 8 0.0 \
#      --memory 0 \
#      --log_dir ${OUTDIR}/${LOGNAME} \
#      --mode ${mode}
#  date
#done

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size     20 for fixed prompt size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name CODAPrompt \
#    --prompt_param 100 8 0.0 \
#    --log_dir ${OUTDIR}/coda


# PATCH-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size     20 for fixed prompt size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#    --prompt_param 21 8 0.0 \
#    --oracle_flag --upper_bound_flag \
# -mtl
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name PATCHPrompt \
#    --prompt_param 21 2 0.0 \
#    --oracle_flag --upper_bound_flag \
#    --log_dir ${OUTDIR}/coda-cond-ip-FPS21-normalattn-oracle-threshold_6-epoch30-mtl

# CODA-P-COND
#
# prompt parameter args:
#    arg 1 = prompt component pool size     20 for fixed prompt size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#    --prompt_param 21 8 0.0 \
#    --oracle_flag --upper_bound_flag \
# -mtl
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name CODAPromptCond \
#    --prompt_param 21 8 0.0 \
#    --log_dir ${OUTDIR}/coda-cond-FPS21-normalattn-oracle-epoch5-cheating-first30-1

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name DualPrompt \
#    --prompt_param 10 20 6 \
#    --log_dir ${OUTDIR}/dual-prompt

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name L2P \
#    --prompt_param 30 20 -1 \
#    --log_dir ${OUTDIR}/l2p++
