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
REPEAT=3
OVERWRITE=0

###############################################################

# process inputsz
mkdir -p $OUTDIR
#mkdir -p ${OUTDIR}/${LOGNAME}/runlog
#    > ${OUTDIR}/${LOGNAME}/runlog/runlog_learn_slot_${time}.out 2>&1

# SLOT-Prompt
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = num of slots extracted from one img
#    arg 4 = num of iter to extract slots
#    arg 5 = temperature to control how sharp are slot attns
#    arg 6 = temperature to control slot selection
#    arg 7 = coeff for weights reg
#    arg 8 = coeff for ccl
#    arg 9 = margin for ccl
#    arg 10 = tau for ccl
#    arg 11 = temperature for cross attn
#    arg 12 = coeff for mk loss
#    --oracle_flag --upper_bound_flag \
#    --debug_mode 1 \
#slot_lrs=(1e-4); temps=(80)
#mk_coeffs=(0.5); slot_vsI_coeffs=(0.5)
#devices=(1); i=-1
#for slot_run_id in 0; do
#for temp_run_id in 0; do
#for mk_coeff_run_id in 0; do
#for slot_vsI_coeff_run_id in 0; do
#((i++))
#slot_lr=${slot_lrs[${slot_run_id}]}
#temp=${temps[${temp_run_id}]}
#mk_coeff=${mk_coeffs[${mk_coeff_run_id}]}
#slot_vsI_coeff=${slot_vsI_coeffs[${slot_vsI_coeff_run_id}]}
#device=${devices[${i}]}
#LOGNAME=4-slot_attn-pos-k10-nt5-recon_noLN-mk${mk_coeff}-crosssim${temp}-slot_vsI${slot_vsI_coeff}-slot_lr${slot_lr}
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 30 40 10 5 1.0 1.0 0.0 0.0 0.1 1.2 ${temp} ${mk_coeff} ${slot_vsI_coeff} \
#    --slot_lr ${slot_lr} \
#    --only_learn_slot \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done
#done
#done
#done

lrs=(2e-4); temps=(10)
selection_ortho_coeffs=(1)
devices=(3); i=-1
for lr_run_id in 0; do
for temp_run_id in 0; do
for soc_run_id in 0; do
((i++))
lr=${lrs[${lr_run_id}]}
temp=${temps[${temp_run_id}]}
selection_ortho_coeff=${selection_ortho_coeffs[${soc_run_id}]}
device=${devices[${i}]}
LOGNAME=5-slot_prompt-k10-nt5-ln-discrete_selec-cossim${temp}-sol${selection_ortho_coeff}-p30-l40-lr${lr}
docker run -d --rm --runtime=nvidia --gpus device=${device} \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/workspace/.cache \
  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type slotmo --learner_name SLOTPrompt \
    --prompt_param 30 40 10 5 1.0 ${temp} 0.0 0.0 0.1 1.2 80 0.5 0.0 ${selection_ortho_coeff} \
    --slot_pre_learn_model 4-slot_attn-pos-k10-nt5-recon_noLN-mk0.5-crosssim80-slot_vsI0.5-slot_lr1e-4 \
    --lr ${lr} ${lr} \
    --log_dir ${OUTDIR}/${LOGNAME}
done
done
done
#    --t0_model_from slot-k10-p30-ccl0-l2weight0.05 \



# PMO-Prompt
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#    arg 4 = num of objectives: 2
#    arg 5 = mask: 0.0; -10000: randn; -10001: uniform; -10002: ortho; -10003: None
#    arg 6 = mask_mode: 0: maskout or 1: use
#    arg 7 = hv coeff, -1 to use LCQP
#    --oracle_flag --upper_bound_flag \
#LEARNERTYPE=pmo
#LEARNERNAME=PMOPrompt
#LOGNAME=pmo-f4m-epoch30-first30-min2
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type ${LEARNERTYPE} --learner_name ${LEARNERNAME} \
#    --prompt_param 21 2 0.0 2 -10003 1 -1 \
#    --log_dir ${OUTDIR}/${LOGNAME}
#date

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size     20 for fixed prompt size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name CODAPrompt \
#    --prompt_param 100 40 0.0 \
#    --lr 0.001 \
#    --log_dir ${OUTDIR}/coda-imagenet-l40

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
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
#python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name L2P \
#    --prompt_param 10 10 -1 \
#    --lr 0.001 \
#    --log_dir ${OUTDIR}/l2p++-imagenet-p10-l10
