# bash experiments/imagenet-r.sh
# experiment settings
DATASET=COBJ
N_CLASS=30

# save directory
OUTDIR=outputs/${DATASET}/3-task

# hard coded inputs
GPUID='0 1'   # '0 1 2 3'
CONFIG=configs/cobj_prompt.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# CODA-P-Replay
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
LEARNERNAME=CODAPromptR
LOGNAME=coda-p-r
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name ${LEARNERNAME} \
    --prompt_param 100 8 0.0 \
    --memory 2000 \
    --log_dir ${OUTDIR}/${LOGNAME}
date

for mode in sys pro non noc
do
  python -u run_ft.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
      --learner_type prompt --learner_name ${LEARNERNAME} \
      --prompt_param 100 8 0.0 \
      --memory 2000 \
      --log_dir ${OUTDIR}/${LOGNAME} \
      --mode ${mode}
  date
done
