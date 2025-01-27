# bash experiments/imagenet-r.sh
# experiment settings
DATASET=COBJ
N_CLASS=30

# save directory
OUTDIR=outputs/${DATASET}/20-10-task

# hard coded inputs
GPUID='0'   # '0 1 2 3'
CONFIG_SLOT=configs/cobj_slot_20-10task.yaml
CONFIG=configs/cobj_prompt_20-10task.yaml
REPEAT=3
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
LOGNAME=coda-l8-p100
docker run -d --rm --runtime=nvidia --gpus device=0 \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/workspace/.cache \
  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 0 \
    --lr 0.001 \
    --max_task 2 \
    --compositional_testing \
    --log_dir ${OUTDIR}/${LOGNAME}

# larger-prompt-lr
LOGNAME=coda-l8-p100-lpl-lr1e-3
docker run -d --rm --runtime=nvidia --gpus device=1 \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/workspace/.cache \
  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
   --learner_type prompt --learner_name CODAPrompt \
   --prompt_param 100 8 0.0 0 \
   --lr 1e-3 \
   --larger_prompt_lr \
   --max_task 2 \
   --compositional_testing \
   --log_dir ${OUTDIR}/${LOGNAME}

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
LEARNERNAME=DualPrompt
LOGNAME=dual-prompt-imagenet-p10-e8-g8
docker run -d --rm --runtime=nvidia --gpus device=2 \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/workspace/.cache \
  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name ${LEARNERNAME} \
    --prompt_param 10 8 8 \
    --lr 0.001 \
    --max_task 2 \
    --compositional_testing \
    --log_dir ${OUTDIR}/${LOGNAME}

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
LEARNERNAME=L2P
LOGNAME=l2p++-imagenet-p10-l8
docker run -d --rm --runtime=nvidia --gpus device=3 \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/workspace/.cache \
  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name ${LEARNERNAME} \
    --prompt_param 10 8 -1 \
    --lr 0.001 \
    --max_task 2 \
    --compositional_testing \
    --log_dir ${OUTDIR}/${LOGNAME}
