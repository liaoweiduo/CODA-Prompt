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
REPEAT=1
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

# concept similar reg
devices=(4 5); i=-1
for concept_similar_reg_coeff in 0.1 1
do
((i++))
device=${devices[${i}]}
LOGNAME=coda-l8-p100-csrc${concept_similar_reg_coeff}
docker run -d --rm --runtime=nvidia --gpus device=${device} \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/workspace/.cache \
  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
   --learner_type prompt --learner_name CODAPrompt \
   --prompt_param 100 8 0.0 0 \
   --lr 0.001 \
   --concept_weight \
   --concept_similar_reg_coeff ${concept_similar_reg_coeff} \
   --eval_class_wise \
   --log_dir ${OUTDIR}/${LOGNAME}
done

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
#
#lr=0.0001
#devices=(2 3); i=-1
#for lr_decreace_ratio in 0.5 1.0
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
#docker run -d --rm --runtime=nvidia --gpus device=3 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name DualPrompt \
#    --prompt_param 10 40 10 \
#    --lr 0.001 \
#    --log_dir ${OUTDIR}/dual-prompt-imagenet-e40-g10


# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
#docker run -d --rm --runtime=nvidia --gpus device=4 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name L2P \
#    --prompt_param 10 40 -1 \
#    --lr 0.001 \
#    --log_dir ${OUTDIR}/l2p++-imagenet-p10-l40


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
#
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