# bash experiments/imagenet-r.sh
# experiment settings
DATASET=CGQA
N_CLASS=100

# save directory
OUTDIR=outputs/${DATASET}/50-10-task

# hard coded inputs
GPUID='0'   # '0 1 2 3'
CONFIG_SLOT=configs/cgqa_slot_50-10task.yaml
CONFIG=configs/cgqa_prompt_50-10task.yaml
REPEAT=5
OVERWRITE=0

###############################################################

# process inputsz
mkdir -p $OUTDIR
#mkdir -p ${OUTDIR}/${LOGNAME}/runlog
#    > ${OUTDIR}/${LOGNAME}/runlog/runlog_learn_slot_${time}.out 2>&1

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size     20 for fixed prompt size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#    arg 4 = 1 FPS
# --oracle_flag --upper_bound_flag \
# -d

LOGNAME=coda-p
#docker run -d --rm --runtime=nvidia --gpus device=1 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
   --learner_type prompt --learner_name CODAPrompt \
   --prompt_param 100 8 0.0 0 \
   --lr 0.001 \
   --do_not_eval_during_training \
   --compositional_testing \
   --log_dir ${OUTDIR}/${LOGNAME}

# random init coda baseline
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
   --learner_type prompt --learner_name CODAPrompt \
   --prompt_param 100 8 0.0 2 \
   --lr 0.001 \
   --compositional_testing \
   --log_dir ${OUTDIR}/coda-p-randint




## concept similar reg + FPS + lr decay
#devices=(0 1 2 3 4 5); i=-1
#for concept_similar_reg_coeff in 0; do
#for lr in 1e-4; do
#for lr_decreace_ratio in 0.1 0.3 0.4 0.6 0.7 0.9; do
#((i++))
#device=${devices[${i}]}
#concept_similar_reg_coeff_sensitivity=0
#LOGNAME=coda-l8-p100-FPS-dcsrc${concept_similar_reg_coeff}_${concept_similar_reg_coeff_sensitivity}-lrd${lr}_${lr_decreace_ratio}
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#   --learner_type prompt --learner_name CODAPrompt \
#   --prompt_param 100 8 0.0 1 \
#   --lr ${lr} \
#   --lr_decreace_ratio ${lr_decreace_ratio} \
#   --concept_weight \
#   --concept_similar_reg_coeff ${concept_similar_reg_coeff} \
#   --concept_similar_reg_coeff_sensitivity ${concept_similar_reg_coeff_sensitivity} \
#   --eval_class_wise \
#   --log_dir ${OUTDIR}/${LOGNAME}
#done
#done
#done

## FPS with lr decrease ratio
#lr=0.001
#devices=(0 1); i=-1
#for lr_decreace_ratio in 0.1 0.5
#do
#((i++))
#device=${devices[${i}]}
#LOGNAME=coda-l8-p100-FPS-lr${lr}_lrd${lr_decreace_ratio}
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#   --learner_type prompt --learner_name CODAPrompt \
#   --prompt_param 100 8 0.0 1 \
#   --lr ${lr} \
#   --lr_decreace_ratio ${lr_decreace_ratio} \
#   --log_dir ${OUTDIR}/${LOGNAME}
#done
##   --eval_class_wise \
##   --concept_weight \
##   --concept_similar_reg_coeff 0.01 \

## larger prompt size
#lr=1e-3
#LOGNAME=coda-l8-p100-lpl-lr${lr}
#docker run -d --rm --runtime=nvidia --gpus device=0 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#   --learner_type prompt --learner_name CODAPrompt \
#   --prompt_param 100 8 0.0 0 \
#   --lr ${lr} \
#   --larger_prompt_lr \
#   --max_task 2 \
#   --compositional_testing \
#   --log_dir ${OUTDIR}/${LOGNAME}
#
## cfst
#for mode in sys pro sub non noc
#do
##  docker run --rm --runtime=nvidia --gpus device=${device} \
##    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
##    -v ~/.cache:/workspace/.cache \
##    --shm-size 8G liaoweiduo/hide:2.0 \
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name CODAPrompt \
#      --prompt_param 100 8 0.0 0 \
#      --log_dir ${OUTDIR}/${LOGNAME} \
#      --mode ${mode}
#  date
#done

## finish other runs
#REPEAT=3
##docker run --rm --runtime=nvidia --gpus device=${device} \
## -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
## -v ~/.cache:/workspace/.cache \
## --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#   --learner_type prompt --learner_name CODAPrompt \
#   --prompt_param 100 8 0.0 0 \
#   --lr 0.001 \
#   --eval_class_wise \
#   --log_dir ${OUTDIR}/${LOGNAME}


# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
#docker run -d --rm --runtime=nvidia --gpus device=2 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 6 20 6 \
    --lr 0.001 \
    --do_not_eval_during_training \
    --compositional_testing \
    --log_dir ${OUTDIR}/dual-prompt


# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
#docker run -d --rm --runtime=nvidia --gpus device=3 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name L2P \
    --prompt_param 30 20 -1 \
    --lr 0.001 \
    --do_not_eval_during_training \
    --compositional_testing \
    --log_dir ${OUTDIR}/l2p++

# vit-pretrain
#
#device=3
#docker run --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name Prompt \
#    --prompt_param 10 10 -1 \
#    --eval_class_wise \
#    --log_dir ${OUTDIR}/vit_pretrain_SGD
##    --oracle_flag --upper_bound_flag \

## cfst
#for mode in sys pro sub non noc
#do
#  docker run --rm --runtime=nvidia --gpus device=${device} \
#    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#    -v ~/.cache:/workspace/.cache \
#    --shm-size 8G liaoweiduo/hide:2.0 \
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name Prompt \
#      --prompt_param 10 10 -1 \
#      --log_dir ${OUTDIR}/vit_pretrain_SGD \
#      --mode ${mode}
#  date
#done