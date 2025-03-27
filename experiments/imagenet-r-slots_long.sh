# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# save directory
OUTDIR=outputs/${DATASET}/100-10-task

# hard coded inputs
GPUID='0'   # '0 1 2 3'
CONFIG_SLOT=configs/imnet-r_slot_long.yaml
REPEAT=1
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
#    --oracle_flag --upper_bound_flag \
#    --debug_mode 1 \

# co-learn slot and prompt  --  Teacher
lr=1e-3
slot_lr1=1e-4
slot_lr2=1e-5

n_slots=4
n_iters=5

#for intra_consistency_reg_coeff in 0 0.1 1; do
intra_consistency_reg_coeff=0.5    # 0.5
intra_consistency_reg_mode=map+cos+kl

slot_ortho_reg_mode=cos+ce
#for slot_ortho_reg_coeff in 0.1 0.5 1 2; do
slot_ortho_reg_coeff=0.5
slot_ortho_reg_temp=1   # dot用0.1

s2p_mode=attn+soft     # sig or soft
#for s2p_temp in $3 $4; do
s2p_temp=10
# soft-temp10, sig-temp1

#slot_logit_similar_reg_mode=map+cos+kl
#slot_logit_similar_reg_coeff=$3
#slot_logit_similar_reg_temp=$4
#slot_logit_similar_reg_slot_temp=1

# bs 256
LOGNAME=rebuttal-k${n_slots}-nt${n_iters}-slot-icr${intra_consistency_reg_coeff}_${intra_consistency_reg_mode}-sor${slot_ortho_reg_coeff}_${slot_ortho_reg_mode}_t${slot_ortho_reg_temp}-s2p_m${s2p_mode}_t${s2p_temp}-slr${slot_lr1}_${slot_lr2}-lr${lr}-p100-l8
#LOGNAME=40-slot-icr${intra_consistency_reg_coeff}_${intra_consistency_reg_mode}-sor${slot_ortho_reg_coeff}_${slot_ortho_reg_mode}_t${slot_ortho_reg_temp}-s2p_m${s2p_mode}_t${s2p_temp}-cheating-slsrc${slot_logit_similar_reg_coeff}_m${slot_logit_similar_reg_mode}_old_t${slot_logit_similar_reg_temp}_${slot_logit_similar_reg_slot_temp}-slr${slot_lr1}_${slot_lr2}-lr${lr}-p100-l8-k10-nt5
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type slotmo --learner_name SLOTPrompt \
    --prompt_param 100 8 \
    --n_slots ${n_slots} \
    --n_iters ${n_iters} \
    --batch_size 256 \
    --s2p_mode ${s2p_mode} \
    --s2p_temp ${s2p_temp} \
    --lr ${lr} ${lr} \
    --slot_lr ${slot_lr1} ${slot_lr2} \
    --use_intra_consistency_reg \
    --intra_consistency_reg_coeff ${intra_consistency_reg_coeff} \
    --intra_consistency_reg_mode ${intra_consistency_reg_mode} \
    --use_slot_ortho_reg \
    --slot_ortho_reg_mode ${slot_ortho_reg_mode} \
    --slot_ortho_reg_coeff ${slot_ortho_reg_coeff}\
    --slot_ortho_reg_temp ${slot_ortho_reg_temp} \
    --max_task 11 \
    --compositional_testing \
    --log_dir ${OUTDIR}/${LOGNAME}
#done
#    --larger_prompt_lr \
#    --concept_weight \
#    --concept_similar_reg_coeff ${concept_similar_reg_coeff} \
#    --concept_similar_reg_temp ${concept_similar_reg_temp} \
#    --use_old_samples_for_reg \
#    --use_slot_logit_similar_reg \
#    --slot_logit_similar_reg_mode ${slot_logit_similar_reg_mode} \
#    --slot_logit_similar_reg_coeff ${slot_logit_similar_reg_coeff} \
#    --slot_logit_similar_reg_temp ${slot_logit_similar_reg_temp} \
#    --slot_logit_similar_reg_slot_temp ${slot_logit_similar_reg_slot_temp} \
#    --use_old_samples_for_reg_no_grad \
#    --eval_class_wise \
#    --oracle_flag --upper_bound_flag \
#    --max_task 1 \

### learn prompt and classifier  -- student
#lr=1e-3
#slot_lr1=1e-4
#slot_lr2=1e-5
#
#intra_consistency_reg_coeff=0.5
#intra_consistency_reg_mode=map+cos+kl
#
#slot_ortho_reg_mode=cos+ce
##for slot_ortho_reg_coeff in 0.1 0.5 1 2; do
#slot_ortho_reg_coeff=0.5
#slot_ortho_reg_temp=1   # dot用0.1
#
#s2p_mode=attn+soft     # sig or soft
##for s2p_temp in $3 $4; do
#s2p_temp=10
#
##for slot_logit_similar_reg_coeff in 0 1; do
#slot_logit_similar_reg_coeff=$1
#slot_logit_similar_reg_mode=map+cos+kl
#for slot_logit_similar_reg_temp in 0.001 0.01; do
##slot_logit_similar_reg_temp=$4    # 0.001
##for slot_logit_similar_reg_slot_temp in 1; do
#slot_logit_similar_reg_slot_temp=1
#
## concept_similar_reg_mode=dot+kl
#SLOT_LOGNAME=44-slot-icr${intra_consistency_reg_coeff}_${intra_consistency_reg_mode}-sor${slot_ortho_reg_coeff}_${slot_ortho_reg_mode}_t${slot_ortho_reg_temp}-s2p_m${s2p_mode}_t${s2p_temp}-slr${slot_lr1}_${slot_lr2}-lr${lr}-p100-l8-k10-nt5
#LOGNAME=44-prompt-cheating-slsrc${slot_logit_similar_reg_coeff}_${slot_logit_similar_reg_mode}_old_t${slot_logit_similar_reg_temp}_${slot_logit_similar_reg_slot_temp}-lr${lr}-icr${intra_consistency_reg_coeff}_${intra_consistency_reg_mode}-sor${slot_ortho_reg_coeff}_${slot_ortho_reg_mode}-s2p_${s2p_mode}_t${s2p_temp}-slr${slot_lr1}_${slot_lr2}-p100-l8-k10-nt5
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 \
#    --batch_size 128 \
#    --s2p_mode ${s2p_mode} \
#    --s2p_temp ${s2p_temp} \
#    --lr ${lr} ${lr} \
#    --slot_pre_learn_model ${SLOT_LOGNAME} \
#    --use_old_samples_for_reg \
#    --use_slot_logit_similar_reg \
#    --slot_logit_similar_reg_mode ${slot_logit_similar_reg_mode} \
#    --slot_logit_similar_reg_coeff ${slot_logit_similar_reg_coeff} \
#    --slot_logit_similar_reg_temp ${slot_logit_similar_reg_temp} \
#    --slot_logit_similar_reg_slot_temp ${slot_logit_similar_reg_slot_temp} \
#    --max_task 6 \
#    --compositional_testing \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done


## separate learn slot and prompt
#lr=1e-3
#slot_lr1=1e-4
#slot_lr2=1e-5
#
#intra_consistency_reg_coeff=$1   # learn 0.1， cross 0.5
#intra_consistency_reg_mode=map+cos+kl
#
#slot_ortho_reg_coeff=$2     # 0.5 or 1
#slot_ortho_reg_mode=cos+ce
#
#s2p_mode=attn+sig    # attn+sig   + soft
#s2p_temp=1       # soft=10, sig=1
#
#SLOT_LOGNAME=40-slot_attn-icr${intra_consistency_reg_coeff}_${intra_consistency_reg_mode}-sor${slot_ortho_reg_coeff}_${slot_ortho_reg_mode}-s2p_${s2p_mode}_t${s2p_temp}-slr${slot_lr1}_${slot_lr2}-pos-k10-nt5-recon_noLN
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 \
#    --batch_size 512 \
#    --s2p_mode ${s2p_mode} \
#    --s2p_temp ${s2p_temp} \
#    --only_learn_slot \
#    --slot_lr ${slot_lr1} ${slot_lr2} \
#    --use_intra_consistency_reg \
#    --intra_consistency_reg_coeff ${intra_consistency_reg_coeff} \
#    --intra_consistency_reg_mode ${intra_consistency_reg_mode} \
#    --use_slot_ortho_reg \
#    --slot_ortho_reg_coeff ${slot_ortho_reg_coeff} \
#    --slot_ortho_reg_mode ${slot_ortho_reg_mode} \
#    --max_task 2 \
#    --log_dir ${OUTDIR}/${SLOT_LOGNAME}
##    --slot_ortho_reg_temp ${slot_ortho_reg_temp} \
#
#slot_logit_similar_reg_coeff=$3
##for slot_logit_similar_reg_coeff in 0 1; do
#slot_logit_similar_reg_mode=map+cos+kl
#slot_logit_similar_reg_temp=$4    # 0.001
#slot_logit_similar_reg_slot_temp=1
#
## concept_similar_reg_mode=dot+kl
#
#LOGNAME=40-slot_prompt-cheating-slsrc${slot_logit_similar_reg_coeff}_${slot_logit_similar_reg_mode}_old_t${slot_logit_similar_reg_temp}_${slot_logit_similar_reg_slot_temp}-lr${lr}-icr${intra_consistency_reg_coeff}_${intra_consistency_reg_mode}-sor${slot_ortho_reg_coeff}_${slot_ortho_reg_mode}-s2p_${s2p_mode}_t${s2p_temp}-slr${slot_lr1}_${slot_lr2}-p100-l8-k10-nt5
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 \
#    --batch_size 128 \
#    --s2p_mode ${s2p_mode} \
#    --s2p_temp ${s2p_temp} \
#    --lr ${lr} ${lr} \
#    --slot_pre_learn_model ${SLOT_LOGNAME} \
#    --use_old_samples_for_reg \
#    --use_slot_logit_similar_reg \
#    --slot_logit_similar_reg_mode ${slot_logit_similar_reg_mode} \
#    --slot_logit_similar_reg_coeff ${slot_logit_similar_reg_coeff} \
#    --slot_logit_similar_reg_temp ${slot_logit_similar_reg_temp} \
#    --slot_logit_similar_reg_slot_temp ${slot_logit_similar_reg_slot_temp} \
#    --max_task 2 \
#    --compositional_testing \
#    --log_dir ${OUTDIR}/${LOGNAME}
##done
##    --slot_pre_learn_model ${SLOT_LOGNAME} \
##    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
##    --concept_weight \
##    --concept_similar_reg_mode ${concept_similar_reg_mode}\
##    --concept_similar_reg_coeff ${concept_similar_reg_coeff} \
##    --concept_similar_reg_temp ${concept_similar_reg_temp} \
##    --t0_model_from 8-slot_prompt-p100-l40-k10-nt5-ln-wA-sigmoid-old5-only_fix_P-cossim10-l1-sol1-dilate1-pcac0.5-lr1e-3 \

## collect class statistics
#lr=1e-3
#temp=1
#device=0
#LOGNAME=6-slot_prompt-sMT-lpl-csrc0.0_s1-lr1e-3-p100-l8-k10-nt5-sig1_FPS
##  -d
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 10 5 1.0 ${temp} 1 0.0 0.0 80 0.0 0.0 0.0 0.0 \
#    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
#    --lr ${lr} ${lr} \
#    --larger_prompt_lr \
#    --use_feature_statistics \
#    --use_slot_statistics \
#    --eval_class_wise \
#    --log_dir ${OUTDIR}/${LOGNAME}

## slot_logit_similar reg + larger prompt lr
#devices=(0 1 2 3 4 5); i=-1
#for slot_logit_similar_reg_coeff in 0.0 0.001 0.01; do
#for lr in 1e-3; do
#temp=1
#slot_logit_similar_reg_coeff_sensitivity=0
#slot_logit_similar_reg_mode=cos+ce
#LOGNAME=7-slot_prompt-sMT-lpl-old_${slot_logit_similar_reg_mode}_slsrc${slot_logit_similar_reg_coeff}_s${slot_logit_similar_reg_coeff_sensitivity}-lr${lr}-p100-l8-k10-nt5-sig${temp}_FPS
#((i++))
#device=${devices[${i}]}
##  -d
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 10 5 1.0 ${temp} 1 0.0 0.0 80 0.0 0.0 0.0 0.0 \
#    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
#    --lr ${lr} ${lr} \
#    --larger_prompt_lr \
#    --use_slot_logit_similar_reg \
#    --use_old_samples_for_reg \
#    --slot_logit_similar_reg_coeff ${slot_logit_similar_reg_coeff} \
#    --slot_logit_similar_reg_coeff_sensitivity ${slot_logit_similar_reg_coeff_sensitivity} \
#    --slot_logit_similar_reg_mode ${slot_logit_similar_reg_mode} \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done
#done
##    --eval_class_wise \

## concept similar reg + larger prompt lr
#concept_similar_reg_coeff=$1
## $1
#for concept_similar_reg_temp in 0.01; do
#concept_similar_reg_mode=dot+ce
#lr=1e-3
#LOGNAME=11-slot_prompt-sMT-cheating-csrc${concept_similar_reg_coeff}_old_${concept_similar_reg_mode}_t${concept_similar_reg_temp}-lr${lr}-p100-l8-k10-nt5-sig1_FPS
##((i++))
##device=${devices[${i}]}
##docker run -d --rm --runtime=nvidia --gpus device=${device} \
##  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
##  -v ~/.cache:/workspace/.cache \
##  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 \
#    --lr ${lr} ${lr} \
#    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
#    --concept_weight \
#    --use_old_samples_for_reg \
#    --concept_similar_reg_coeff ${concept_similar_reg_coeff} \
#    --concept_similar_reg_mode ${concept_similar_reg_mode} \
#    --concept_similar_reg_temp ${concept_similar_reg_temp} \
#    --max_task 3 \
#    --compositional_testing \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done
##    --larger_prompt_lr \
##    --use_old_samples_for_reg_no_grad \
##    --eval_class_wise \

## cfst
#for mode in sys pro sub non noc
#do
# # do not use -d to avoid running in parallel
##  docker run --rm --runtime=nvidia --gpus device=${device} \
##    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
##    -v ~/.cache:/workspace/.cache \
##    --shm-size 8G liaoweiduo/hide:2.0 \
#  python -u run_ft.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 \
#    --s2p_temp ${temp} \
#    --log_dir ${OUTDIR}/${LOGNAME} \
#    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
#    --lr 0.001 \
#    --use_feature_statistics \
#    --mode ${mode}
#  date
#done

# finish other runs
#REPEAT=3
#docker run --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 10 5 1.0 ${temp} 2 0.0 0.0 80 0.0 0.0 0.0 0.0 \
#    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
#    --lr ${lr} ${lr} \
#    --log_dir ${OUTDIR}/${LOGNAME}


## try mlp s2p mode
#devices=(1 2 3); i=-1
#for weight_coeff in 0.01 0.05 0.1
#do
#((i++))
#lr=1e-4
#device=${devices[${i}]}
#LOGNAME=5-slot_prompt-sMT-p100-l8-k10-nt5-s2p_mlp_FPS-pllr${lr}-wc${weight_coeff}
##  -d
#docker run -d --rm --runtime=nvidia --gpus device=${device} \
#  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
#  -v ~/.cache:/workspace/.cache \
#  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 10 5 1.0 1.0 3 ${weight_coeff} 0.0 80 0.0 0.0 0.0 0.0 \
#    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
#    --lr ${lr} ${lr} \
#    --eval_class_wise \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done
