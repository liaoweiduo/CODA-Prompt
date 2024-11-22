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

# process inputs
mkdir -p $OUTDIR


# SLOT-Prompt
#for mode in sys pro sub non noc
#do
#  # do not use -d to avoid running in parallel
#  docker run --rm --runtime=nvidia --gpus device=6 \
#    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#    -v ~/.cache:/workspace/.cache \
#    --shm-size 8G liaoweiduo/hide:2.0 \
#  python -u run_ft.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 40 10 5 1.0 10 0.0 0.0 0.0 0.0 80 0.0 0.0 0.0 0.0 \
#    --log_dir ${OUTDIR}/MT-slot_prompt-p100-l40-k10-nt5-ln-wA-sigmoid-onehotl1-cossim10-l1_sol1-dilate1_contrast_cos_pcac0.1-lr1e-3 \
#    --slot_pre_learn_model 4-slot_attn-pos-k10-nt5-recon_noLN-mk0.5-crosssim80-slot_vsI0.5-slot_lr1e-4 \
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
for mode in sys pro sub non noc
do
  docker run --rm --runtime=nvidia --gpus device=6 \
    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
    -v ~/.cache:/workspace/.cache \
    --shm-size 8G liaoweiduo/hide:2.0 \
  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
      --learner_type prompt --learner_name CODAPrompt \
      --prompt_param 1 40 0.0 \
      --log_dir ${OUTDIR}/MT-one-prompt-imagenet-l40-lr1e-3 \
      --mode ${mode}
  date
done

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
#for mode in sys pro sub non noc
#do
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
#for mode in sys pro sub non noc
#do
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name L2P \
#      --prompt_param 10 10 -1 \
#      --log_dir ${OUTDIR}/l2p++-imagenet-p10-l10 \
#      --mode ${mode}
#  date
#done

# vit-pretrain
#
#for mode in sys pro sub non noc
#do
#  docker run --rm --runtime=nvidia --gpus device=6 \
#    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#    -v ~/.cache:/workspace/.cache \
#    --shm-size 8G liaoweiduo/hide:2.0 \
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name Prompt \
#      --prompt_param 10 10 -1 \
#      --log_dir ${OUTDIR}/vit_pretrain \
#      --mode ${mode}
#  date
#done
##      --eval_every_epoch \