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
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR


# SLOT-Prompt
#for mode in sys pro non noc
#do
#  # do not use -d to avoid running in parallel
#  docker run --rm --runtime=nvidia --gpus device=4 \
#    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#    -v ~/.cache:/workspace/.cache \
#    --shm-size 8G liaoweiduo/hide:2.0 \
#  python -u run_ft.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 30 40 10 5 1.0 10 0.0 0.0 0.1 1.2 80 0.5 0.5 1 \
#    --log_dir ${OUTDIR}/5-slot_prompt-k10-nt5-ln-discrete_selec-cossim10-sol1-p30-l40-lr2e-4 \
#    --slot_pre_learn_model 4-slot_attn-pos-k10-nt5-recon_noLN-mk0.5-crosssim80-slot_vsI0.5-slot_lr1e-5 \
#    --lr 0.001 \
#    --mode ${mode}
#  date
#done


# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#for mode in sys pro non noc
#do
#  # do not use -d to avoid running in parallel
#  docker run --rm --runtime=nvidia --gpus device=4 \
#    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#    -v ~/.cache:/workspace/.cache \
#    --shm-size 8G liaoweiduo/hide:2.0 \
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name CODAPrompt \
#      --prompt_param 100 40 0.0 \
#      --log_dir ${OUTDIR}/coda-imagenet-l40 \
#      --mode ${mode}
#  date
#done

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
#for mode in sys pro non noc
#do
#  # do not use -d to avoid running in parallel
#  docker run --rm --runtime=nvidia --gpus device=4 \
#    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#    -v ~/.cache:/workspace/.cache \
#    --shm-size 8G liaoweiduo/hide:2.0 \
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name DualPrompt \
#      --prompt_param 10 40 10 \
#      --log_dir ${OUTDIR}/dual-prompt-imagenet-e40-g10 \
#      --mode ${mode}
#  date
#done

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
for mode in sys pro non noc
do
#  # do not use -d to avoid running in parallel
#  docker run --rm --runtime=nvidia --gpus device=4 \
#    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#    -v ~/.cache:/workspace/.cache \
#    --shm-size 8G liaoweiduo/hide:2.0 \
  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
      --learner_type prompt --learner_name L2P \
      --prompt_param 10 10 -1 \
      --log_dir ${OUTDIR}/l2p++-imagenet-p10-l10 \
      --mode ${mode}
  date
done