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
# --oracle_flag --upper_bound_flag \
# -d
LOGNAME=coda
#device=1
#docker run --rm --runtime=nvidia --gpus device=${device} \
# -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
# -v ~/.cache:/workspace/.cache \
# --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#   --learner_type prompt --learner_name CODAPrompt \
#   --prompt_param 100 8 0.0 0 \
#   --lr 0.001 \
#   --do_not_eval_during_training \
#   --compositional_testing \
#   --log_dir ${OUTDIR}/${LOGNAME}

# random init coda baseline
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#   --learner_type prompt --learner_name CODAPrompt \
#   --prompt_param 100 8 0.0 2 \
#   --lr 0.001 \
#   --compositional_testing \
#   --log_dir ${OUTDIR}/coda-p-randint

# mapping from feature->slot for prompt selection baseline
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#   --learner_type prompt --learner_name CODAPrompt \
#   --prompt_param 100 8 0.0 3 \
#   --lr 0.001 \
#   --compositional_testing \
#   --log_dir ${OUTDIR}/coda-p-f2s_linear

# directly use concept to reg logits
for concept_similar_reg_coeff in $1 $2 $3; do
#concept_similar_reg_coeff=0.1
concept_similar_reg_mode=dot+kl
concept_similar_reg_temp=0.01    # 5 for cos
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
   --learner_type prompt --learner_name CODAPrompt \
   --prompt_param 100 8 0.0 0 \
   --lr 0.001 \
   --concept_weight \
   --concept_similar_reg_coeff ${concept_similar_reg_coeff} \
   --concept_similar_reg_mode ${concept_similar_reg_mode} \
   --concept_similar_reg_temp ${concept_similar_reg_temp} \
   --compositional_testing \
   --max_task 2 \
   --log_dir ${OUTDIR}/coda-p-concept-${concept_similar_reg_coeff}
done

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
#    --prompt_param 10 20 6 \
#    --lr 0.001 \
#    --do_not_eval_during_training \
#    --compositional_testing \
#    --log_dir ${OUTDIR}/dual-prompt


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
#    --prompt_param 30 20 -1 \
#    --lr 0.001 \
#    --do_not_eval_during_training \
#    --compositional_testing \
#    --log_dir ${OUTDIR}/l2p++


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