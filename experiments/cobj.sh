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
#slot_lrs=(0.0002 0.0003 0.0004 0.0006)
#devices=(4 5 6 7)
#for run_id in 0 1 2 3; do
#slot_lr=${slot_lrs[${run_id}]}
#device=${devices[${run_id}]}
#LOGNAME=slot-k10-recon-slot_lr${slot_lr}
#time=$(date +"%y-%m-%d-%H-%M-%S-%N")
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  --shm-size 8G liaoweiduo/coda:2.0_sklearn \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
#    --prompt_param 8 10 0.05 1.0 0.1 1.2 \
#    --slot_lr ${slot_lr} \
#    --only_learn_slot \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done

ccl_coeffs=(0.01 0.04 0.07 0.1)
devices=(4 5 6 7)
for run_id in 0 1 2 3; do
ccl_coeff=${ccl_coeffs[${run_id}]}
device=${devices[${run_id}]}
LOGNAME=slot-k10-ccl${ccl_coeff}-l2weight0.05
docker run -d --rm --runtime=nvidia --gpus device=${device} \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  --shm-size 8G liaoweiduo/coda:2.0_sklearn \
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
    --prompt_param 8 10 0.05 ${ccl_coeff} 0.1 1.2 \
    --slot_pre_learn_model slot-k10-recon-slot_lr0.0002 \
    --t0_model_from slot-k10-ccl0-l2weight0.05 \
    --log_dir ${OUTDIR}/${LOGNAME}
done


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
