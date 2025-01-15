# bash experiments/imagenet-r.sh
# experiment settings
DATASET=CGQA
N_CLASS=100

# save directory
OUTDIR=outputs/${DATASET}/50-10-task

# hard coded inputs
GPUID='0'   # '0 1 2 3'
CONFIG_SLOT=configs/cgqa_slot_50-10task.yaml
CONFIG=configs/cgqa_prompt_50-10task.yaml
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

# co-learn slot and prompt
# $1
lr=1e-3
slot_lr1=5e-4
slot_lr2=1e-4
intra_consistency_reg_coeff=$1
intra_consistency_reg_mode=learn+l2

slot_ortho_reg_coeff=1.0
slot_ortho_reg_temp=0.1

for slot_logit_similar_reg_coeff in 0 0.01; do
slot_logit_similar_reg_temp=0.01
slot_logit_similar_reg_slot_temp=0.1

LOGNAME=16-slot-icr${intra_consistency_reg_coeff}_m${intra_consistency_reg_mode}-sor${slot_ortho_reg_coeff}_t${slot_ortho_reg_temp}-cheating-slsrc${slot_logit_similar_reg_coeff}_old_t${slot_logit_similar_reg_temp}_${slot_logit_similar_reg_slot_temp}-lr${lr}-p100-l8-k10-nt5-sig1_FPS
#LOGNAME=15-slot-icr${intra_consistency_reg_coeff}_m${intra_consistency_reg_mode}-lr${lr}-p100-l8-k10-nt5-sig1_FPS
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type slotmo --learner_name SLOTPrompt \
    --prompt_param 100 8 \
    --batch_size 128 \
    --lr ${lr} ${lr} \
    --slot_lr ${slot_lr1} ${slot_lr2} \
    --use_intra_consistency_reg \
    --intra_consistency_reg_coeff ${intra_consistency_reg_coeff} \
    --intra_consistency_reg_mode ${intra_consistency_reg_mode} \
    --use_slot_ortho_reg \
    --slot_ortho_reg_coeff ${slot_ortho_reg_coeff}\
    --slot_ortho_reg_temp ${slot_ortho_reg_temp} \
    --use_old_samples_for_reg \
    --use_slot_logit_similar_reg \
    --slot_logit_similar_reg_coeff ${slot_logit_similar_reg_coeff} \
    --slot_logit_similar_reg_temp ${slot_logit_similar_reg_temp} \
    --slot_logit_similar_reg_slot_temp ${slot_logit_similar_reg_slot_temp} \
    --max_task 3 \
    --compositional_testing \
    --log_dir ${OUTDIR}/${LOGNAME}
done
#    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
#    --larger_prompt_lr \
#    --concept_weight \
#    --concept_similar_reg_coeff ${concept_similar_reg_coeff} \
#    --concept_similar_reg_temp ${concept_similar_reg_temp} \
#    --use_old_samples_for_reg_no_grad \
#    --eval_class_wise \

## separate learn slot and prompt
#lr=1e-3
#slot_lr1=5e-4
#slot_lr2=1e-4
#intra_consistency_reg_coeff=0.01
#intra_consistency_reg_mode=cross+l1
##slot_ortho_reg_temp=0.5
#slot_ortho_reg_mode=l2
#slot_ortho_reg_coeff=0.5
##for slot_ortho_reg_coeff in 0.1 0.5 1.0; do
#SLOT_LOGNAME=slot_attn-pos-k10-nt5-recon_noLN-icr${intra_consistency_reg_coeff}_m${intra_consistency_reg_mode}-sor${slot_ortho_reg_coeff}_m${slot_ortho_reg_mode}-slr${slot_lr1}_${slot_lr2}
##docker run -d --rm --runtime=nvidia --gpus device=${device} \
##  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
##  -v ~/.cache:/workspace/.cache \
##  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 \
#    --batch_size 256 \
#    --only_learn_slot \
#    --slot_lr ${slot_lr1} ${slot_lr2} \
#    --use_intra_consistency_reg \
#    --intra_consistency_reg_coeff ${intra_consistency_reg_coeff} \
#    --intra_consistency_reg_mode ${intra_consistency_reg_mode} \
#    --use_slot_ortho_reg \
#    --slot_ortho_reg_coeff ${slot_ortho_reg_coeff} \
#    --slot_ortho_reg_mode ${slot_ortho_reg_mode} \
#    --max_task 3 \
#    --log_dir ${OUTDIR}/${SLOT_LOGNAME}
##    --slot_ortho_reg_temp ${slot_ortho_reg_temp} \
#
##concept_similar_reg_coeff=1.0
##concept_similar_reg_temp=0.01
#LOGNAME=16-slot_prompt-icr${intra_consistency_reg_coeff}_m${intra_consistency_reg_mode}-sor${slot_ortho_reg_coeff}_m${slot_ortho_reg_mode}-slr${slot_lr1}_${slot_lr2}-lr${lr}-p100-l8-k10-nt5-sig1_FPS
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 \
#    --batch_size 128 \
#    --lr ${lr} ${lr} \
#    --slot_pre_learn_model ${SLOT_LOGNAME} \
#    --max_task 3 \
#    --compositional_testing \
#    --log_dir ${OUTDIR}/${LOGNAME}
##done
###    --t0_model_from 8-slot_prompt-p100-l40-k10-nt5-ln-wA-sigmoid-old5-only_fix_P-cossim10-l1-sol1-dilate1-pcac0.5-lr1e-3 \

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
