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
slot_lrs=(1e-4 2e-4); temps=(1)
devices=(3 4 5); i=-1
for slot_run_id in 0 1; do
for temp_run_id in 0; do
((i++))
slot_lr=${slot_lrs[${slot_run_id}]}
temp=${temps[${temp_run_id}]}
device=${devices[${i}]}
LOGNAME=2-slot_attn-pos-k10-nt5-temp${temp}-recon_noLN-mk-slot_lr${slot_lr}
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


#lrs=(1e-4); temps=(5 11 31)
#devices=(0 1 2); i=-1
#for lr_run_id in 0; do
#for temp_run_id in 0 1 2; do
#((i++))
#lr=${lrs[${lr_run_id}]}
#temp=${temps[${temp_run_id}]}
#device=${devices[${i}]}
#LOGNAME=1-slot_prompt-minmax-k10-nt5-ptemp${temp}-p30-l40-lr${lr}
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 30 40 10 5 1.0 ${temp} 0.0 0.0 0.1 1.2 \
#    --slot_pre_learn_model 1-slot_attn-pos-k10-nt5-temp1-recon_noLN-slot_lr1e-4 \
#    --lr ${lr} ${lr} \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done
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
