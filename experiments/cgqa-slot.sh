# bash experiments/imagenet-r.sh
# experiment settings
DATASET=CGQA
N_CLASS=100

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0'   # '0 1 2 3'
CONFIG=configs/cgqa_slot.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# Slot
#
# prompt parameter args:
#    arg 1 = slot size
#    arg 2 = pen dim size
#    --oracle_flag --upper_bound_flag \
LEARNERTYPE=decoder
LEARNERNAME=SLOT
LOGNAME=slot-lr4e-4
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
    --prompt_param 5 21 \
    --log_dir ${OUTDIR}/${LOGNAME}
date
