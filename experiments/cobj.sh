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
REPEAT=3
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# SLOT-Prompt
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = num of slots extracted from one img
#    --debug_mode 1 \
LEARNERTYPE=slotmo
LEARNERNAME=SLOTPrompt
#slot_lrs=(0.0001)
#devices=(4)
#for run_id in 0; do
#slot_lr=${slot_lrs[${run_id}]}
#device=${devices[${run_id}]}
LOGNAME=slot-k10-recon-mk-SGD-slot_lr5e-3
##time=$(date +"%y-%m-%d-%H-%M-%S-%N")
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
    --prompt_param 30 40 10 0.0 0.0 0.1 1.2 \
    --slot_lr 5e-3 \
    --only_learn_slot \
    --log_dir ${OUTDIR}/${LOGNAME}
#done

#lrs=(0.00001 0.00005 0.0001 0.0002)
#devices=(4 5 6 7)
#for run_id in 0 1 2 3; do
#lr=${lrs[${run_id}]}
#device=${devices[${run_id}]}
#LOGNAME=slot-k10-coda-p100-l40-lr${lr}
##LOGNAME=slot-k10-coda-p30-mk-ccl${ccl_coeff}-l2weight0.05
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
#    --prompt_param 100 40 10 0.0 0.0 0.1 1.2 \
#    --slot_pre_learn_model slot-k10-recon-mk-slot_lr0.0001 \
#    --lr ${lr} ${lr} \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done
##    --t0_model_from slot-k10-p30-ccl0-l2weight0.05 \


# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#LEARNERNAME=CODAPrompt
#LOGNAME=coda-imagenet-l40
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 100 40 0.0 \
#    --log_dir ${OUTDIR}/${LOGNAME}

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
#LEARNERNAME=DualPrompt
#LOGNAME=dual-prompt-imagenet-e40-g10
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 10 40 10 \
#    --log_dir ${OUTDIR}/${LOGNAME}

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
#LEARNERNAME=L2P
#LOGNAME=l2p++-imagenet-p10-l10
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 10 10 -1 \
#    --log_dir ${OUTDIR}/${LOGNAME}
