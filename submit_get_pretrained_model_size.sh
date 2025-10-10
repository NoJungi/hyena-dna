#!/bin/bash
#SBATCH --job-name=tiny
#SBATCH --output=slurm/get_pretrained_model_params/test_fine_tune_tiny_%j.out
#SBATCH --error=slurm/get_pretrained_model_params/test_fine_tune_tiny_%j.err
#SBATCH --nodes=1
#SBATCH --partition=vision
#SBATCH --gpus=1
#SBATCH --time=01:00:00    #10:00:00
#SBATCH --mem=100gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

PRETRAINED_MODEL="hyenadna-tiny-1k-seqlen-d256" # "hyenadna-tiny-1k-seqlen" 
N_LAYER=2
D_MODEL=256
L_MAX=1026
ORDER=2

FREEZE=false # or true  
DECODER="CNN_BEND"
LR=0.00006
BATCH_SIZE=128 #32
EPOCHS=1 #50

# make name for output dir
LR_name=$(echo $LR | awk -F. '{print $2}') # only digits after '.'
# 'freeze' if FREEZE is True else 'fully_tuned'
FREEZE_NAME=$(if [ "$FREEZE" = true ]; then echo "freeze"; else echo "fully_tuned"; fi)
PRETRAINED_MODEL_PATH="/home/s-nojung/Masterarbeit/Code/hyena-dna/pretrained_weights/${PRETRAINED_MODEL}/weights.ckpt"

DATE=$(date '+%Y-%m-%d_%H-%M-%S')
# Dynamischen Run-Namen und Ordner bauen
RUN_NAME="${FREEZE_NAME}_${BATCH_SIZE}_batch_size_${EPOCHS}_epochs_${DATE}"
OUT_DIR="./outputs/fine_tuning/CCE_F1_loss/${DECODER}_${PRETRAINED_MODEL}_${LR_name}_lr/${RUN_NAME}"
WANDB_RUN_NAME="$fine_tuning_F1_loss_${DECODER}_${PRETRAINED_MODEL}_${FREEZE_NAME}_${BATCH_SIZE}_batch_size_${EPOCHS}_epochs_${LR_name}_lr"


echo "start fine tuning with:"
echo "  pretrained_model   = ${PRETRAINED_MODEL}"
echo "  d_model            = ${D_MODEL}"
echo "  l_max              = ${L_MAX}"
echo "  n_layer            = ${N_LAYER}"
echo "  order              = ${ORDER}"
echo "  freeze             = ${FREEZE}"
echo "  decoder            = ${DECODER}"
echo "  lr                 = ${LR}"
echo "  batch              = ${BATCH_SIZE}"
echo "  epochs             = ${EPOCHS}"
echo "  Output             = ${OUT_DIR}"
echo "  wandb Run-Name     = ${WANDB_RUN_NAME}"

# Training starten mit Hydra-Overrides
time python -m train experiment=hg38/gene_finding \
dataset.max_length=$L_MAX \
dataset.batch_size=$BATCH_SIZE \
model.d_model=$D_MODEL \
model.n_layer=$N_LAYER \
model.layer.order=$ORDER \
decoder._name_=$DECODER \
trainer.max_epochs=$EPOCHS \
optimizer.lr=$LR \
wandb.name=$WANDB_RUN_NAME \
hydra.run.dir=$OUT_DIR \
train.pretrained_model_path=$PRETRAINED_MODEL_PATH \
train.pretrained_model_state_hook.freeze_backbone=$FREEZE \
wandb.mode="disabled"

sstat -j "$SLURM_JOB_ID".batch --format=JobID,MaxVMSize
