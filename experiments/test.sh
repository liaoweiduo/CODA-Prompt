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

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name PMOPrompt \
    --prompt_param 100 8 0.0 2000 \
    --log_dir ${OUTDIR}/debug \
    --debug_mode 1

rm data

# for PyCharm run.py
#--config
#configs/cgqa_prompt.yaml
#--gpuid
#0
#--repeat
#1
#--overwrite
#0
#--learner_type
#prompt
#--learner_name
#PMOPrompt
#--prompt_param
#100
#8
#0.0
#2000
#--memory
#0
#--log_dir
#../CODA-Prompt-experiments/CGQA/10-task/debug
#--debug_mode
#1
#--dataroot
#"../../../OneDrive - City University of Hong Kong - Student/datasets"