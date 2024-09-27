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
REPEAT=3
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR
#mkdir -p ${OUTDIR}/${LOGNAME}/runlog
#    > ${OUTDIR}/${LOGNAME}/runlog/runlog_learn_slot_${time}.out 2>&1

# SLOT-Prompt
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = num of slots extracted from one img
#    arg 4 = num of iter to extract slots
#    arg 5 = temperature to control how sharp are slot attns
#    arg 6 = temperature to control slot selection
#    arg 7 = coeff for weights reg
#    arg 8 = coeff for ccl
#    arg 9 = margin for ccl
#    arg 10 = tau for ccl
#    --oracle_flag --upper_bound_flag \
#    --debug_mode 1 \
slot_lrs=(1e-4 2e-4); temps=(0.5 1 2)
devices=(0 1 2 3 4 5 6 7 8); i=-1
for slot_run_id in 0 1; do
for temp_run_id in 0 1 2; do
((i++))
slot_lr=${slot_lrs[${slot_run_id}]}
temp=${temps[${temp_run_id}]}
device=${devices[${i}]}
LOGNAME=1-slot_attn-pos-k10-nt5-temp${temp}-recon_noLN-slot_lr${slot_lr}
#slot_lr=1e-4
#temp=5
#LOGNAME=1-slot_attn-pos-k10-nt5-temp${temp}-recon_noLN-slot_lr${slot_lr}
docker run -d --rm --runtime=nvidia --gpus device=${device} \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/workspace/.cache \
  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type slotmo --learner_name SLOTPrompt \
    --prompt_param 30 40 10 5 ${temp} 1.0 0.0 0.0 0.1 1.2 \
    --slot_lr ${slot_lr} \
    --only_learn_slot \
    --log_dir ${OUTDIR}/${LOGNAME}
done
done

#lrs=(0.0001)
#devices=(6)
#for run_id in 0; do   # 0 1 2 3
#lr=${lrs[${run_id}]}
#device=${devices[${run_id}]}
#LOGNAME=1-slot_prompt-k10-nt5-ptemp1-p30-l40-lr${lr}
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 30 40 10 5 1.0 1.0 0.0 0.0 0.1 1.2 \
#    --slot_pre_learn_model 1-slot_attn-k10-nt5-temp1-recon_noLN-slot_lr1e-4 \
#    --lr ${lr} ${lr} \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done
##    --t0_model_from slot-k10-p30-ccl0-l2weight0.05 \


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
#    --prompt_param 100 40 0.0 \
#    --log_dir ${OUTDIR}/coda-imagenet-l40

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name DualPrompt \
#    --prompt_param 10 40 10 \
#    --log_dir ${OUTDIR}/dual-prompt-imagenet-e40-g10

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name L2P \
#    --prompt_param 10 10 -1 \
#    --log_dir ${OUTDIR}/l2p++-imagenet-p10-l10
