#!/bin/bash
#SBATCH --job-name=train_256
#SBATCH --output=slurm/train_full_CCE_loss_CNN_BEND_1026_max_length_00006_lr_256_d_model_3_order_32_batch_size_50_epochs_%a_layers_%A.out
#SBATCH --error=slurm/train_full_CCE_loss_CNN_BEND_1026_max_length_00006_lr_256_d_model_3_order_32_batch_size_50_epochs_%a_layers_%A.err
#SBATCH --nodes=1
#SBATCH --partition=vision
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem=100gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --array=1-8

D_MODEL=256
N_LAYER=${SLURM_ARRAY_TASK_ID}
ORDER=3     # min: 2

MAX_LENGTH=1026
DECODER="CNN_BEND"
LR=0.00006
BATCH_SIZE=32
EPOCHS=50

# make name for output dir
LR_NAME=$(echo $LR | awk -F. '{print $2}') # only digits after '.'

DATE=$(date '+%Y-%m-%d_%H-%M-%S')
# defined output directory and name for the wandb logging
RUN_NAME="${ORDER}_order_${BATCH_SIZE}_batch_size_${EPOCHS}_epochs_${DATE}"
OUT_DIR="./outputs/train_full/CCE_loss/${DECODER}_${MAX_LENGTH}_max_length_${LR_NAME}_lr_${D_MODEL}_d_model/${N_LAYER}_layers/${RUN_NAME}"
WANDB_RUN_NAME="train_full_CCE_loss_${DECODER}_${N_LAYER}_layers_${ORDER}_order_${MAX_LENGTH}_max_len_${D_MODEL}_d_model_${BATCH_SIZE}_batch_size_${EPOCHS}_epochs_${LR_NAME}_lr"

echo "start training with:"
echo "  d_model            = ${D_MODEL}"
echo "  n_layer            = ${N_LAYER}"
echo "  order              = ${ORDER}"
echo "  max_len            = ${MAX_LENGTH}"
echo "  decoder            = ${DECODER}"
echo "  lr                 = ${LR}"
echo "  batch              = ${BATCH_SIZE}"
echo "  epochs             = ${EPOCHS}"
echo "  Run-Name           = ${RUN_NAME}"
echo "  Output             = ${OUT_DIR}"

# Start training with Hydra overrides
time python -m train experiment=hg38/BEND_gene_finding_cce \
dataset.max_length=$MAX_LENGTH \
dataset.batch_size=$BATCH_SIZE \
model.d_model=$D_MODEL \
model.n_layer=$N_LAYER \
model.layer.order=$ORDER \
decoder._name_=$DECODER \
trainer.max_epochs=$EPOCHS \
optimizer.lr=$LR \
wandb.name=$WANDB_RUN_NAME \
hydra.run.dir=$OUT_DIR

sstat -j "$SLURM_JOB_ID".batch --format=JobID,MaxVMSize
