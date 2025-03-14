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

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size     20 for fixed prompt size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
# --oracle_flag --upper_bound_flag \
# -d
LOGNAME=coda-l8-p100
device=1
docker run --rm --runtime=nvidia --gpus device=${device} \
 -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
 -v ~/.cache:/workspace/.cache \
 --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
   --learner_type prompt --learner_name CODAPrompt \
   --prompt_param 100 8 0.0 \
   --lr 0.001 \
   --eval_class_wise \
   --log_dir ${OUTDIR}/${LOGNAME}

# cfst
for mode in sys pro sub non noc
do
  docker run --rm --runtime=nvidia --gpus device=${device} \
    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
    -v ~/.cache:/workspace/.cache \
    --shm-size 8G liaoweiduo/hide:2.0 \
  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
      --learner_type prompt --learner_name CODAPrompt \
      --prompt_param 100 8 0.0 \
      --log_dir ${OUTDIR}/${LOGNAME} \
      --mode ${mode}
  date
done

LOGNAME=coda-l8-p21
docker run --rm --runtime=nvidia --gpus device=${device} \
 -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
 -v ~/.cache:/workspace/.cache \
 --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
   --learner_type prompt --learner_name CODAPrompt \
   --prompt_param 21 8 0.0 \
   --lr 0.001 \
   --eval_class_wise \
   --log_dir ${OUTDIR}/${LOGNAME}

# cfst
for mode in sys pro sub non noc
do
  docker run --rm --runtime=nvidia --gpus device=${device} \
    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
    -v ~/.cache:/workspace/.cache \
    --shm-size 8G liaoweiduo/hide:2.0 \
  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
      --learner_type prompt --learner_name CODAPrompt \
      --prompt_param 21 8 0.0 \
      --log_dir ${OUTDIR}/${LOGNAME} \
      --mode ${mode}
  date
done

LOGNAME=coda-l8-p100
# finish other runs
REPEAT=3
docker run --rm --runtime=nvidia --gpus device=${device} \
 -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
 -v ~/.cache:/workspace/.cache \
 --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
   --learner_type prompt --learner_name CODAPrompt \
   --prompt_param 100 8 0.0 \
   --lr 0.001 \
   --eval_class_wise \
   --log_dir ${OUTDIR}/${LOGNAME}

LOGNAME=coda-l8-p21
# finish other runs
REPEAT=3
docker run --rm --runtime=nvidia --gpus device=${device} \
 -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
 -v ~/.cache:/workspace/.cache \
 --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
   --learner_type prompt --learner_name CODAPrompt \
   --prompt_param 21 8 0.0 \
   --lr 0.001 \
   --eval_class_wise \
   --log_dir ${OUTDIR}/${LOGNAME}


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
#docker run -d --rm --runtime=nvidia --gpus device=0 \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name Prompt \
#    --prompt_param 10 10 -1 \
#    --eval_class_wise \
#    --oracle_flag --upper_bound_flag \
#    --log_dir ${OUTDIR}/MT-vit_pretrain