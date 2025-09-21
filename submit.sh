#!/bin/bash
#SBATCH --job-name=train_3_4
#SBATCH --output=slurm/train_full_CNN_BEND_1026_max_length_00006_lr_128_d_model_3_layer_4_order_32_batch_size_100_epochs_no_overlap_%j.out
#SBATCH --error=slurm/train_full_CNN_BEND_1026_max_length_00006_lr_128_d_model_3_layer_4_order_32_batch_size_100_epochs_no_overlap_%j.err
#SBATCH --nodes=1
#SBATCH --partition=vision
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=100gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32


D_MODEL=128
N_LAYER=3
ORDER=4     # min: 2


MAX_LENGTH=1026
DECODER="CNN_BEND"
LR=0.00006
BATCH_SIZE=32
EPOCHS=100
LAST_CHUNK_OVERLAP=false  # True or False


# make name for output dir
LR_name=$(echo $LR | awk -F. '{print $2}') # only digits after '.'

if [ "$LAST_CHUNK_OVERLAP" = false ] ; then
  LAST_CHUNK_OVERLAP_name="no_overlap"
else
  LAST_CHUNK_OVERLAP_name="with_overlap"
fi 

DATE=$(date '+%Y-%m-%d_%H-%M-%S')
# Dynamischen Run-Namen und Ordner bauen
RUN_NAME="${ORDER}_order_${BATCH_SIZE}_batch_size_${EPOCHS}_epochs_${LAST_CHUNK_OVERLAP_name}_${DATE}"
OUT_DIR="./outputs/train_full/${DECODER}_${MAX_LENGTH}_max_length_${LR_name}_lr_${D_MODEL}_d_model_/${N_LAYER}_layers/${RUN_NAME}"
WANDB_RUN_NAME="$train_full_${DECODER}_${N_LAYER}_layers_${ORDER}_order_${MAX_LENGTH}_max_len_${D_MODEL}_d_model_${BATCH_SIZE}_batch_size_${LAST_CHUNK_OVERLAP_name}_${EPOCHS}_epochs_${LR_name}_lr"


echo "start training with:"
echo "  d_model            = ${D_MODEL}"
echo "  n_layer            = ${N_LAYER}"
echo "  order              = ${ORDER}"
echo "  max_len            = ${MAX_LENGTH}"
echo "  last_chunk_overlap = ${LAST_CHUNK_OVERLAP}"
echo "  decoder            = ${DECODER}"
echo "  lr                 = ${LR}"
echo "  batch              = ${BATCH_SIZE}"
echo "  epochs             = ${EPOCHS}"
echo "  Run-Name           = ${RUN_NAME}"
echo "  Output             = ${OUT_DIR}"

# Training starten mit Hydra-Overrides
time python -m train experiment=hg38/gene_finding \
dataset.max_length=$MAX_LENGTH \
dataset.batch_size=$BATCH_SIZE \
dataset.last_chunk_overlap=$LAST_CHUNK_OVERLAP \
model.d_model=$D_MODEL \
model.n_layer=$N_LAYER \
model.layer.order=$ORDER \
decoder._name_=$DECODER \
trainer.max_epochs=$EPOCHS \
optimizer.lr=$LR \
wandb.name=$WANDB_RUN_NAME \
hydra.run.dir=$OUT_DIR

sstat -j "$SLURM_JOB_ID".batch --format=JobID,MaxVMSize
