# bash experiments/imagenet-r.sh
# experiment settings
DATASET=COBJ
N_CLASS=30

# save directory
OUTDIR=outputs/${DATASET}/3-task

# hard coded inputs
GPUID='0'   # '0 1 2 3'
CONFIG_SLOT=configs/cobj_slot.yaml
CONFIG=configs/cobj_prompt.yaml
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
LEARNERTYPE=slotmo
LEARNERNAME=SLOTPrompt
for slot_lr in 0.0001 0.0005 0.001 0.005
do
LOGNAME=slot-k10-recon-slot_lr${slot_lr}
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
    --prompt_param 100 8 10 0.05 \
    --slot_lr ${slot_lr} \
    --only_learn_slot \
    --log_dir ${OUTDIR}/${LOGNAME}
date
done
#LOGNAME=slot-k10-l2weight-coeff0.05
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
#    --prompt_param 100 8 10 0.05 \
#    --slot_lr ${slot_lr} \
#    --slot_pre_learn_model slot-k10-recon-slot_lrxxxx \
#    --log_dir ${OUTDIR}/${LOGNAME}

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#LEARNERNAME=CODAPrompt
#LOGNAME=coda-p
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 100 8 0.0 \
#    --log_dir ${OUTDIR}/${LOGNAME}

#for mode in sys pro non noc
#do
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name ${LEARNERNAME} \
#      --prompt_param 100 8 0.0 \
#      --log_dir ${OUTDIR}/${LOGNAME} \
#      --mode ${mode}
#  date
#done

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
#LEARNERNAME=DualPrompt
#LOGNAME=dual-prompt
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 10 20 6 \
#    --log_dir ${OUTDIR}/${LOGNAME}

#for mode in sys pro non noc
#do
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name ${LEARNERNAME} \
#      --prompt_param 10 20 6 \
#      --log_dir ${OUTDIR}/${LOGNAME} \
#      --mode ${mode}
#  date
#done


# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
#LEARNERNAME=L2P
#LOGNAME=l2p++
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 30 20 -1 \
#    --log_dir ${OUTDIR}/${LOGNAME}

#for mode in sys pro non noc
#do
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name ${LEARNERNAME} \
#      --prompt_param 30 20 -1 \
#      --log_dir ${OUTDIR}/${LOGNAME} \
#      --mode ${mode}
#  date
#done
