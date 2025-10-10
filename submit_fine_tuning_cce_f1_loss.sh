#!/bin/bash
#SBATCH --job-name=tiny_128
#SBATCH --output=slurm/fine_tuning_CNN_BEND_tiny_128_d_model_fully_tuned_00006_lr_32_batch_size_50_epochs_%j.out
#SBATCH --error=slurm/fine_tuning_CNN_BEND_tiny_128_d_model_fully_tuned_00006_lr_32_batch_size_50_epochs_%j.err
#SBATCH --nodes=1
#SBATCH --partition=vision
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem=100gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

PRETRAINED_MODEL="hyenadna-tiny-1k-seqlen" # "hyenadna-tiny-1k-seqlen-d256" 
D_MODEL=128 #128
FREEZE=false #true  

DECODER="CNN_BEND"
LR=0.00006
BATCH_SIZE=32
EPOCHS=50

# make name for output dir
LR_NAME=$(echo $LR | awk -F. '{print $2}') # only digits after '.'
# 'freeze' if FREEZE is True else 'fully_tuned'
FREEZE_NAME=$(if [ "$FREEZE" = true ]; then echo "freeze"; else echo "fully_tuned"; fi)
PRETRAINED_MODEL_PATH="/home/s-nojung/Masterarbeit/Code/hyena-dna/pretrained_weights/${PRETRAINED_MODEL}/weights.ckpt"

DATE=$(date '+%Y-%m-%d_%H-%M-%S')
# name for output dir and wandb logging
RUN_NAME="${FREEZE_NAME}_${BATCH_SIZE}_batch_size_${EPOCHS}_epochs_${DATE}"
OUT_DIR="./outputs/fine_tuning/CCE_F1_loss/${DECODER}_${PRETRAINED_MODEL}_${LR_NAME}_lr/${RUN_NAME}"
WANDB_RUN_NAME="fine_tuning_F1_loss_${DECODER}_${PRETRAINED_MODEL}_${FREEZE_NAME}_${BATCH_SIZE}_batch_size_${EPOCHS}_epochs_${LR_NAME}_lr"

echo "start fine tuning with:"
echo "  pretrained_model   = ${PRETRAINED_MODEL}"
echo "  d_model            = ${D_MODEL}"
echo "  freeze             = ${FREEZE}"
echo "  decoder            = ${DECODER}"
echo "  lr                 = ${LR}"
echo "  batch              = ${BATCH_SIZE}"
echo "  epochs             = ${EPOCHS}"
echo "  Output             = ${OUT_DIR}"
echo "  wandb Run-Name     = ${WANDB_RUN_NAME}"

# Start training with Hydra overrides
time python -m train experiment=hg38/BEND_gene_finding_cce_f1_loss \
dataset.batch_size=$BATCH_SIZE \
model.d_model=$D_MODEL \
decoder._name_=$DECODER \
trainer.max_epochs=$EPOCHS \
optimizer.lr=$LR \
wandb.name=$WANDB_RUN_NAME \
hydra.run.dir=$OUT_DIR \
train.pretrained_model_path=$PRETRAINED_MODEL_PATH \
train.pretrained_model_state_hook.freeze_backbone=$FREEZE

sstat -j "$SLURM_JOB_ID".batch --format=JobID,MaxVMSize
