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

# process inputsz
mkdir -p $OUTDIR
#mkdir -p ${OUTDIR}/${LOGNAME}/runlog
#    > ${OUTDIR}/${LOGNAME}/runlog/runlog_learn_slot_${time}.out 2>&1

# PMO-Prompt
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#    --oracle_flag --upper_bound_flag \
#LOGNAME=pmo-concept_w2_1-1p-l8
##docker run -d --rm --runtime=nvidia --gpus device=1 \
##  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
##  -v ~/.cache:/workspace/.cache \
##  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type pmo --learner_name PMOPrompt \
#    --prompt_param 21 8 0.0 \
#    --lr 0.001 \
#    --concept_weight \
#    --eval_class_wise \
#    --oracle_flag --upper_bound_flag \
#    --log_dir ${OUTDIR}/${LOGNAME}
#date
##    --target_concept_id 0 \

