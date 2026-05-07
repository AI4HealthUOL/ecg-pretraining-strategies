#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX6K:1
#SBATCH --nodelist=mpcg007
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=mpcg_smds.p
#SBATCH --time=1-00:00
#SBATCH --output=/dev/null

BASE_DIR="/user/tunu6250/github_repo/ecg-pretraining-strategies/downstream_evaluation"
CHECKPOINTS_DIR="/fs/s6k/groups/agaifh/tmp/fm_eval_ecg/checkpoints"
DATASET_DIR="/fs/s6k/groups/agaifh/datasets/ecg_new"

EVAL_MODE="finetuning_linear"    # finetuning_linear, frozen, linear
MODEL="ecg_jepa_multiblock"                 # data2vec, dinosr, jepa, cpc, hubert_pp, s4, ecg_founder, ecg_jepa_multiblock
DATASET="cpsc_extra"            # ningbo, cpsc2018, cpsc_extra, georgia, chapman, sph, ptbxl_all, ptbxl_sub, ptbxl_super, zzu_pecg, echonext, mimic
LEARNING_RATE=0.001
BATCH_SIZE=64
EPOCHS=100

PREFIX="full_dataset"

if [ "$EVAL_MODE" == "finetuning_linear" ]; then
    OUTPUT_DIR="${BASE_DIR}/${PREFIX}_finetuning_linear/outputs"
    PREDICTIONS_DIR="${BASE_DIR}/${PREFIX}_finetuning_linear/predictions"
    LOGS_DIR="${BASE_DIR}/${PREFIX}_finetuning_linear/logs"
elif [ "$EVAL_MODE" == "frozen" ]; then
    OUTPUT_DIR="${BASE_DIR}/${PREFIX}_frozen/outputs"
    PREDICTIONS_DIR="${BASE_DIR}/${PREFIX}_frozen/predictions"
    LOGS_DIR="${BASE_DIR}/${PREFIX}_frozen/logs"
elif [ "$EVAL_MODE" == "linear" ]; then
    OUTPUT_DIR="${BASE_DIR}/${PREFIX}_linear/outputs"
    PREDICTIONS_DIR="${BASE_DIR}/${PREFIX}_linear/predictions"
    LOGS_DIR="${BASE_DIR}/${PREFIX}_linear/logs"
else
    echo "Error: Unknown mode '$EVAL_MODE'. Choose from finetuning_linear, frozen, or linear."
    exit 1
fi

module load hpc-env/13.1
module load CUDA/12.9.1
module load Anaconda3
module load git
module load GCC/13.1.0

conda activate lightning3_blackwell

mkdir -p "${LOGS_DIR}/${MODEL}"
mkdir -p "${OUTPUT_DIR}/${MODEL}_${DATASET}"
mkdir -p "${PREDICTIONS_DIR}/${MODEL}"

LOG_FILE="${LOGS_DIR}/${MODEL}/${DATASET}_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Special handling per dataset
ARGS_DATASET=()
case $DATASET in
  "ningbo")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ningbo"
        "--fs-data 500"
        "--finetune-dataset ningbo"
    )
    ;;
  "cpsc2018")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/cpsc2018"
        "--fs-data 500"
        "--finetune-dataset cpsc2018"
    )
    ;;
  "cpsc_extra")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/cpsc_extra"
        "--fs-data 500"
        "--finetune-dataset cpsc_extra"
    )
    ;;
  "georgia")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/georgia"
        "--fs-data 500"
        "--finetune-dataset georgia"
    )
    ;;
  "chapman")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/chapman_ECGData"
        "--fs-data 500"
        "--finetune-dataset chapman"
    )
    ;;
  "sph")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/sph"
        "--fs-data 500"
        "--finetune-dataset sph"
    )
    ;;
  "code15_diag")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/code15"
        "--fs-data 400"
        "--finetune-dataset code15_diag"
    )
    ;;
  "ptbxl_super")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptbxl_records500"
        "--fs-data 500"
        "--finetune-dataset ptbxl_super"
    )
    ;;
  "ptbxl_sub")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptbxl_records500"
        "--fs-data 500"
        "--finetune-dataset ptbxl_sub"
    )
    ;;
  "ptbxl_all")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/ptbxl_records500"
        "--fs-data 500"
        "--finetune-dataset ptbxl_all"
    )
    ;;
  "echonext")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/echonext"
        "--fs-data 250"
        "--finetune-dataset echonext"
    )
    ;; 
  "mimic")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/mimic"
        "--fs-data 500"
        "--finetune-dataset mimic"
    )
    ;;
  "zzu_pecg")
    ARGS_DATASET+=(
        "--data ${DATASET_DIR}/zzu_pecg"
        "--fs-data 500"
        "--finetune-dataset zzu_pecg"
    )
    ;;
esac

# Special handling per model
ARGS_MODEL=()
case $MODEL in
  "data2vec")
    ARGS_MODEL+=(
        "--architecture data2vec"
        "--input-size 2.5"
        "--fs-model 240"
        "--input-channels 12"
        "--precision 32"
        "--pretrained ${CHECKPOINTS_DIR}/pretrained/heedb-emory-code15/final/config_data2vec_ecg_s4.yaml"
    )
    ;;
  "dinosr")
    ARGS_MODEL+=(
        "--architecture dinosr"
        "--input-size 2.5"
        "--fs-model 240"
        "--input-channels 12"
        "--precision 32"
        "--pretrained ${CHECKPOINTS_DIR}/pretrained/heedb-emory-code15/final/config_dinosr_ecg_s4.yaml"
    )
    ;;
  "jepa")
    ARGS_MODEL+=(
        "--architecture jepa"
        "--input-size 2.5"
        "--fs-model 240"
        "--input-channels 12"
        "--precision 32"
        "--pretrained ${CHECKPOINTS_DIR}/pretrained/heedb-emory-code15/final/config_jepa_ecg_s4.yaml"
    )
    ;;
  "hubert_pp")
    ARGS_MODEL+=(
        "--architecture hubert_pp"
        "--input-size 2.5"
        "--fs-model 240"
        "--input-channels 12"
        "--precision 32"
        "--pretrained ${CHECKPOINTS_DIR}/pretrained/heedb-emory-code15/final/config_skkmeans_ecg_s4.yaml"
    )
    ;;
  "cpc")
    ARGS_MODEL+=(
        "--architecture cpc"
        "--input-size 2.5"
        "--fs-model 240"
        "--input-channels 12"
        "--precision 32"
        "--pretrained ${CHECKPOINTS_DIR}/pretrained/heedb-emory-code15/final/config_cpc_ecg_s4.yaml"
    )
    ;;
  "s4")
    ARGS_MODEL+=(
        "--architecture s4"
        "--input-size 2.5"
        "--fs-model 100"
        "--input-channels 12"
        "--s4-n 8"
        "--s4-h 512"
        "--s4-layers 4"
        "--precision 32"
    )
    ;;
  "ecg_founder")
    ARGS_MODEL+=(
        "--architecture ecg_founder"
        "--input-size 2.5"
        "--fs-model 500"
        "--input-channels 12"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_founder/12_lead_ECGFounder.pth"
    )
    ;;
  "ecg_jepa_multiblock")
    ARGS_MODEL+=(
        "--architecture ecg_jepa"
        "--input-size 10" 
        "--fs-model 250"
        "--input-channels 8"
        "--pretrained ${CHECKPOINTS_DIR}/ecg_jepa/multiblock_epoch100.pth"
    )
    ;;
esac

# Run the experiment
python ${BASE_DIR}/code/main_lite.py \
  ${ARGS_DATASET[@]} \
  ${ARGS_MODEL[@]} \
  --epochs ${EPOCHS} \
  --modality ecg \
  --lr ${LEARNING_RATE} \
  --batch-size ${BATCH_SIZE} \
  --finetune \
  --eval-mode ${EVAL_MODE} \
  --output-path "${OUTPUT_DIR}/${MODEL}_${DATASET}" \
  --prediction-path "${PREDICTIONS_DIR}/${MODEL}" \
  --export-predictions
