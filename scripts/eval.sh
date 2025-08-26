#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb
#PBS -l walltime=0:20:00
#PBS -o /rds/general/user/mss124/home/thesis/marl_graph_exploration/job_o
#PBS -e /rds/general/user/mss124/home/thesis/marl_graph_exploration/job_e

cd "$PBS_O_WORKDIR"

module purge
module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0
source ~/thesis/marl/bin/activate

echo "Running evaluation in: $(pwd)"

# Base parameters for the environment, matching the training setup
# Note: --device=cuda should be used if the model was trained on GPU, even for eval
BASE_ENV_PARAMS="--env-type=routing --n-router=20 --n-data=30 --env-var=1 --episode-steps=300 --ttl=0 --random-topology=0 --topology-init-seed=324821133 --train-topology-allow-eval-seed --no-congestion --activation-function=leaky_relu --model=dgn --hidden-dim=512,256 --num-heads=8 --num-attention-layers=2 --device=cuda"

# Evaluation specific parameters
EVAL_PARAMS="--eval --eval-episodes=1000 --eval-episode-steps=300 --disable-progressbar --policy=trained"

# Path to your best trained model
MODEL_PATH="/rds/general/user/mss124/home/thesis/marl_graph_exploration/runs/Jun29_13-27-21_cx3-20-1.cx3.hpc.ic.ac.uk_R1_DGN_fixed-dgn-t324821133-intr1/model_best.pt"

# Output directory for evaluation results
DATE=$(date +%Y%m%d_%H%M%S)
EVAL_DIR_NAME="eval_results_${DATE}"
mkdir -p $EVAL_DIR_NAME

# Run the evaluation
(set -x; time python -u src/main.py \
    $BASE_ENV_PARAMS \
    $EVAL_PARAMS \
    --model-load-path=$MODEL_PATH \
    --eval-output-dir=${EVAL_DIR_NAME} \
    --comment="evaluation_of_best_model" \
    ) > ${EVAL_DIR_NAME}/evaluation_log.log 2>&1