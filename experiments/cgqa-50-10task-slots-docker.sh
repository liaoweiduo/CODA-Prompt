

devices=(0 1 2 3 4); i=-1
for concept_similar_reg_temp in 0.005 0.01 0.05 0.1; do
((i++))
device=${devices[${i}]}
docker run -d --rm --runtime=nvidia --gpus device=${device} \
  -v ~/CODA-Prompt:/workspace -v /mnt/datasets/datasets:/workspace/data -v ~/checkpoints:/checkpoints \
  -v ~/.cache:/workspace/.cache \
  --shm-size 8G liaoweiduo/hide:2.0 \
bash experiments/cgqa-50-10task-slots.sh ${concept_similar_reg_temp}
done