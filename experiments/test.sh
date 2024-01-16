# bash experiments/imagenet-r.sh
# experiment settings
DATASET=CGQA
N_CLASS=100

# save directory
OUTDIR=../CODA-Prompt-experiments/${DATASET}/10-task

# hard coded inputs
GPUID='0'
CONFIG=configs/cgqa_prompt.yaml
REPEAT=1
OVERWRITE=0

###############################################################
# ln -s data
rm data
ln -s '../../../OneDrive - City University of Hong Kong - Student/datasets' .
mv datasets data

# process inputs
mkdir -p $OUTDIR

# CODA-P-Replay
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPromptR \
    --prompt_param 100 8 0.0 \
    --memory 100 \
    --log_dir ${OUTDIR}/debug

rm data
