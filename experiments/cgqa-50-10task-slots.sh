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
#devices=(5); i=-1
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
#LOGNAME=0-slot_attn-pos-k10-nt5-recon_noLN-mk${mk_coeff}-crosssim${temp}-slot_vsI${slot_vsI_coeff}-slot_lr${slot_lr}
##docker run -d --rm --runtime=nvidia --gpus device=${device} \
##  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
##  -v ~/.cache:/workspace/.cache \
##  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 30 40 10 5 1.0 1.0 0.0 0.0 ${temp} ${mk_coeff} ${slot_vsI_coeff} \
#    --slot_lr ${slot_lr} \
#    --only_learn_slot \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done
#done
#done
#done

#lrs=(1e-3); temps=(10)
#coeffs=(0)
#devices=(0); i=-1
#for lr_run_id in 0; do
#for temp_run_id in 0; do
#for coef_run_id in 0; do
#((i++))
#lr=${lrs[${lr_run_id}]}
#temp=${temps[${temp_run_id}]}
#coeff=${coeffs[${coef_run_id}]}
#device=${devices[${i}]}
#LOGNAME=2-slot_prompt-img_repr-sMT-p100-l40-k10-nt5-ln-wA-sigmoid-onehotl1-cossim${temp}-l1_sol1-dilate1_contrast_cos_pcac${coeff}-lr${lr}
##docker run -d --rm --runtime=nvidia --gpus device=${device} \
##  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
##  -v ~/.cache:/workspace/.cache \
##  --shm-size 8G liaoweiduo/hide:2.0 \
#python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 40 10 5 1.0 ${temp} 0.0 1.0 80 0.5 0.0 1.0 ${coeff} \
#    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
#    --lr ${lr} ${lr} \
#    --log_dir ${OUTDIR}/${LOGNAME}
#done
#done
#done
##    --t0_model_from 8-slot_prompt-p100-l40-k10-nt5-ln-wA-sigmoid-old5-only_fix_P-cossim10-l1-sol1-dilate1-pcac0.5-lr1e-3 \

# concept similar reg + FPS + lr decay
devices=(0 1 2 3 4 5); i=-1
for concept_similar_reg_coeff in 0.05 0.1 0.15 0.2 0.25 0.3; do
for lr in 1e-4; do
for lr_decreace_ratio in 1.0; do
((i++))
device=${devices[${i}]}
temp=1
LOGNAME=6-slot_prompt-sMT-p100-l8-k10-nt5-FPS-sig${temp}_FPS-dcsrc${concept_similar_reg_coeff}_1-lrd${lr}_${lr_decreace_ratio}
#  -d
docker run -d --rm --runtime=nvidia --gpus device=${device} \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/workspace/.cache \
  --shm-size 8G liaoweiduo/hide:2.0 \
python -u run.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type slotmo --learner_name SLOTPrompt \
    --prompt_param 100 8 10 5 1.0 ${temp} 10 0.0 0.0 80 0.0 0.0 0.0 0.0 \
    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
    --lr ${lr} ${lr} \
    --lr_decreace_ratio ${lr_decreace_ratio} \
    --concept_weight \
    --concept_similar_reg_coeff ${concept_similar_reg_coeff} \
    --eval_class_wise \
    --log_dir ${OUTDIR}/${LOGNAME}
done
done
done

# cfst
#for mode in sys pro sub non noc
#do
# # do not use -d to avoid running in parallel
##  docker run --rm --runtime=nvidia --gpus device=${device} \
##    -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
##    -v ~/.cache:/workspace/.cache \
##    --shm-size 8G liaoweiduo/hide:2.0 \
#  python -u run_ft.py --config $CONFIG_SLOT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type slotmo --learner_name SLOTPrompt \
#    --prompt_param 100 8 10 5 1.0 ${temp} 1 0.0 0.0 80 0.0 0.0 0.0 0.0 \
#    --log_dir ${OUTDIR}/${LOGNAME} \
#    --slot_pre_learn_model MT-slot_attn-pos-k10-nt5-recon_noLN-intra0.01-crosssim10-slot_vsI0.5-slot_lr1e-4 \
#    --lr 0.001 \
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
