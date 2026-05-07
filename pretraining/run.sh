#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX6K:1
#SBATCH --nodelist=mpcg007
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=mpcg_smds.p
#SBATCH --time=1-00:00
#SBATCH --output=/dev/null


###############################################################################
# CONFIGS
###############################################################################
FRAMEWORK="hubert++"                            # data2vec, dinosr, jepa, cpc, hubert++
CONFIG_FILE_NAME="config_skkmeans_ecg_s4.yaml"  # config_data2vec_ecg_s4.yaml, config_dinosr_ecg_s4.yaml, config_jepa_ecg_s4.yaml, config_cpc_ecg_s4.yaml, config_skkmeans_ecg_s4.yaml
###############################################################################

BASE_DIR="/user/tunu6250/github_repo/ecg-pretraining-strategies/pretraining"

module load hpc-env/13.1
module load CUDA/12.9.1
module load Anaconda3
module load git

cd ${BASE_DIR}
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

conda activate lightning3_blackwell

LOGS_DIR="${BASE_DIR}/logs"
OUTPUTS_DIR="${BASE_DIR}/outputs/${FRAMEWORK}_${SLURM_JOB_ID}"
HYDRA_LOGS_DIR="${BASE_DIR}/hydra_logs/${FRAMEWORK}_${SLURM_JOB_ID}"

mkdir -p ${LOGS_DIR}
mkdir -p ${OUTPUTS_DIR}
mkdir -p ${HYDRA_LOGS_DIR}

LOG_FILE="${LOGS_DIR}/${FRAMEWORK}_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

python ${BASE_DIR}/code/main_all.py \
        --config-name=${CONFIG_FILE_NAME} \
        trainer.output_path="${OUTPUTS_DIR}" \
        hydra.run.dir="${HYDRA_LOGS_DIR}" \
        trainer.revision="${COMMIT}"