# bash experiments/imagenet-r.sh
# experiment settings
DATASET=CGQA
N_CLASS=100

# save directory
OUTDIR=outputs/${DATASET}/2-task

# hard coded inputs
GPUID='0'   # '0 1 2 3'
CONFIG_SLOT=configs/cgqa_slot_2task.yaml
CONFIG=configs/cgqa_prompt_2task.yaml
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

LOGNAME_t0=pmo-concept_w1-1st_p1-coda-l8-adam-task0
LOGNAME=pmo-concept_w1-1st_p1-coda-l8-adam
# use target_concept_id = -1 to disable reweighting data
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type pmo --learner_name PMOPrompt \
    --prompt_param 1 8 0.0 \
    --lr 0.001 \
    --max_task 1 \
    --concept_weight \
    --target_concept_id -1 \
    --eval_class_wise \
    --log_dir ${OUTDIR}/${LOGNAME_t0}
date
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type pmo --learner_name PMOPrompt \
    --prompt_param 1 8 0.0 \
    --lr 0.001 \
    --concept_weight \
    --target_concept_id -1 \
    --prompt_pre_learn_mode ${LOGNAME_t0} \
    --eval_class_wise \
    --log_dir ${OUTDIR}/${LOGNAME}
date

for target_concept_id in {0..20}
do
LOGNAME_t0=pmo-concept_w4_.5-1st_p1-coda-l8-adam-concept${target_concept_id}-task0
LOGNAME=pmo-concept_w4_.5-1st_p1-coda-l8-adam-concept${target_concept_id}
#docker run -d --rm --runtime=nvidia --gpus device=1 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type pmo --learner_name PMOPrompt \
    --prompt_param 1 8 0.0 \
    --lr 0.001 \
    --max_task 1 \
    --concept_weight \
    --target_concept_id ${target_concept_id} \
    --eval_class_wise \
    --log_dir ${OUTDIR}/${LOGNAME_t0}
date

#docker run -d --rm --runtime=nvidia --gpus device=0 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type pmo --learner_name PMOPrompt \
    --prompt_param 1 8 0.0 \
    --lr 0.001 \
    --concept_weight \
    --target_concept_id ${target_concept_id} \
    --prompt_pre_learn_mode ${LOGNAME_t0} \
    --eval_class_wise \
    --log_dir ${OUTDIR}/${LOGNAME}
date
done

## FPS
#LOGNAME=pmo-concept_w.9_.1-FPS-1p-l8-concept0
#docker run -d --rm --runtime=nvidia --gpus device=1 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type pmo --learner_name PMOPrompt \
#    --prompt_param 21 8 0.0 \
#    --lr 0.001 \
#    --concept_weight \
#    --target_concept_id 0 \
#    --eval_class_wise \
#    --log_dir ${OUTDIR}/${LOGNAME}
#date

# pmo-1p
#LOGNAME=pmo-1p-l8
#docker run -d --rm --runtime=nvidia --gpus device=2 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type pmo --learner_name PMOPrompt \
#    --prompt_param 1 8 0.0 \
#    --lr 0.001 \
#    --concept_weight \
#    --target_concept_id 0 \
#    --prompt_pre_learn_mode pmo-1p-task0-l8 \
#    --eval_class_wise \
#    --log_dir ${OUTDIR}/${LOGNAME}
#date

## learn selection
#LOGNAME=pmo-selection-concept_w.9_.1-1p-l8
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type pmo --learner_name PMOPrompt \
#    --prompt_param 21 8 0.0 \
#    --lr 0.001 \
#    --prompt_pre_learn_mode pmo-concept_w.9_.1-1p-l8 \
#    --eval_class_wise \
#    --oracle_flag --upper_bound_flag \
#    --log_dir ${OUTDIR}/${LOGNAME}
#date

## cfst
#for mode in sys pro sub non noc
#do
# # do not use -d to avoid running in parallel
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type pmo --learner_name PMOPrompt \
#    --prompt_param 21 8 0.0 \
#    --log_dir ${OUTDIR}/${LOGNAME} \
#    --lr 0.001 \
#    --mode ${mode}
#  date
#done
##    --target_concept_id 0 \

