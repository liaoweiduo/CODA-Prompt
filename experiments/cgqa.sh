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
#mkdir -p ${OUTDIR}/${LOGNAME}/runlog
#    > ${OUTDIR}/${LOGNAME}/runlog/runlog_learn_slot_${time}.out 2>&1

# SLOT-Prompt
#
# prompt parameter args:
#    arg 1 = prompt length
#    arg 2 = num of slots extracted from one img
#    arg 3 = coeff for weights reg
#    arg 4 = coeff for ccl
#    arg 5 = margin for ccl
#    arg 6 = tau for ccl
#    --oracle_flag --upper_bound_flag \
#    --debug_mode 1 \
LEARNERTYPE=slotmo
LEARNERNAME=SLOTPrompt
#slot_lrs=(0.0003)
#devices=(3)   # (0 1 2 3)
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

ccl_coeffs=(0.13 0.17 0.2 0.25)
devices=(0 1 2 3)
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
    --slot_pre_learn_model slot-k10-recon-slot_lr0.0003 \
    --t0_model_from slot-k10-ccl0-l2weight0.05 \
    --log_dir ${OUTDIR}/${LOGNAME}
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
