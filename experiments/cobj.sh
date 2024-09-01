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
#
LEARNERTYPE=slotmo
LEARNERNAME=SLOTPrompt
for coeff in 0.001 0.003 0.005 0.007 0.009
do
LOGNAME=slot-k5-recon-l2weight-coeff${coeff}-lr1e-4
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
    --prompt_param 100 8 5 $coeff 30 \
    --log_dir ${OUTDIR}/${LOGNAME}
date
done

# CODA-P-Replay
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#LEARNERNAME=CODAPromptR
#LOGNAME=coda-p-r
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 100 8 0.0 \
#    --memory 2000 \
#    --log_dir ${OUTDIR}/${LOGNAME}
#date
#
#for mode in sys pro non noc
#do
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name ${LEARNERNAME} \
#      --prompt_param 100 8 0.0 \
#      --memory 2000 \
#      --log_dir ${OUTDIR}/${LOGNAME} \
#      --mode ${mode}
#  date
#done

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#LEARNERNAME=CODAPrompt
#LOGNAME=coda-p
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 100 8 0.0 \
#    --log_dir ${OUTDIR}/${LOGNAME}

#for mode in sys pro non noc
#do
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name ${LEARNERNAME} \
#      --prompt_param 100 8 0.0 \
#      --log_dir ${OUTDIR}/${LOGNAME} \
#      --mode ${mode}
#  date
#done


# CODA-P-COND
#
# prompt parameter args:
#    arg 1 = prompt component pool size     20 for fixed prompt size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#    --prompt_param 21 8 0.0 \
#    --oracle_flag --upper_bound_flag \
# -mtl
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name CODAPromptCond \
#    --prompt_param 20 8 0.0 \
#    --oracle_flag --upper_bound_flag \
#    --log_dir ${OUTDIR}/cobj-coda-cond-FPS20-normalattn-oracle-epoch10-cheating-mtl-1


# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
#LEARNERNAME=DualPrompt
#LOGNAME=dual-prompt
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 10 20 6 \
#    --log_dir ${OUTDIR}/${LOGNAME}
#
#for mode in sys pro non noc
#do
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name ${LEARNERNAME} \
#      --prompt_param 10 20 6 \
#      --log_dir ${OUTDIR}/${LOGNAME} \
#      --mode ${mode}
#  date
#done


# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
#LEARNERNAME=L2P
#LOGNAME=l2p++
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name ${LEARNERNAME} \
#    --prompt_param 30 20 -1 \
#    --log_dir ${OUTDIR}/${LOGNAME}
#
#for mode in sys pro non noc
#do
#  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#      --learner_type prompt --learner_name ${LEARNERNAME} \
#      --prompt_param 30 20 -1 \
#      --log_dir ${OUTDIR}/${LOGNAME} \
#      --mode ${mode}
#  date
#done
