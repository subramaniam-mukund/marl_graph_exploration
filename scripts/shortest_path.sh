#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb
#PBS -l walltime=48:00:00
#PBS -o /rds/general/user/mss124/home/thesis/marl_graph_exploration/job_o
#PBS -e /rds/general/user/mss124/home/thesis/marl_graph_exploration/job_e

# Navigate to the working directory where the script is submitted from
cd "$PBS_O_WORKDIR"

# Purge existing modules and load necessary ones
module purge
module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0

# Activate the Python virtual environment
source ~/thesis/marl/bin/activate

echo "Running in: $(pwd)"

# Define common parameters for the Python script
# BASE_PARAMS contains general settings for the RL environment and model
BASE_PARAMS="--gamma=0.9 --epsilon=1.0 --hidden-dim=512,256 --mini-batch-size=32 --device=cuda --lr=0.001 --tau=0.01 --step-before-train=10_000 --capacity=200_000 --eval-episodes=100 --eval-episode-steps=300 --disable-progressbar"
# LIMITS define training steps and frequency
LIMITS="--step-between-train=10 --total-steps=250_000"

# Generate a timestamp for the main log directory to ensure uniqueness
DATE=$(date +%Y%m%d_%H%M%S)
# Define the main directory name for all logs and results from this batch run
DIR_NAME="shortest_path_experiment_degree4"

# Create the main log directory if it doesn't exist
mkdir -p $DIR_NAME

# Define a helper function to execute a single Python run
# $1: RUN_NAME (for log file and output directory naming)
# $@: RUN_ARGS (all other arguments passed to the Python script)
run() {
    RUN_NAME="$1"
    shift # Shift arguments so $1 is now the first RUN_ARGS
    RUN_ARGS="$@" # Capture all remaining arguments as RUN_ARGS
    # Execute the Python script, redirecting stdout and stderr to a unique log file
    # `set -x` prints commands before execution for debugging
    # `time` measures execution time
    # `-u` for unbuffered output from Python
    (set -x; time python -u src/main.py $RUN_ARGS --comment=${RUN_NAME}) > ${DIR_NAME}/${RUN_NAME}.log 2>&1
}

EVAL_SEEDS=$(python -c "from src.env.constants_degree4 import EVAL_SEEDS")

# Start timing the entire batch execution
time (
for topology_seed in $EVAL_SEEDS
do
    eval_run_seed=0
    # Run the shortest path heuristic evaluation
    # --policy=heuristic: Specifies using the heuristic shortest path policy
    # --eval: Indicates this is an evaluation run, not training
    # --seed=${eval_run_seed}: Sets the random seed for the evaluation process
    # --topology-init-seed=${topology_seed}: Sets the seed for generating the topology
    # --train-topology-allow-eval-seed: Allows the evaluation seed to influence topology if needed (often used with random_topology)
    # --episode-steps=300: Max steps per episode during evaluation
    # --random-topology=0: Ensures a fixed, non-random topology based on topology-init-seed
    # --n-data=30: Specifies the number of data points/episodes for evaluation (from your original script)
    # --disable-progressbar: Suppresses progress bar output
    # --eval-output-dir: Specifies the unique directory for evaluation results for this specific topology
    run "shortest-nocong-paths-eval-topo-${topology_seed}-run-${eval_run_seed}" \
        --policy=heuristic \
        --eval \
        --seed=${eval_run_seed} \
        --no-congestion \
        --topology-init-seed=${topology_seed} \
        --train-topology-allow-eval-seed \
        --episode-steps=300 \
        --random-topology=0 \
        --n-data=20 \
        --n-router=20 \
        --degree=4 \
        --disable-progressbar \
        --eval-output-dir=${DIR_NAME}/eval_results_topo_${topology_seed}_run_${eval_run_seed}

    run "shortest-paths-eval-topo-${topology_seed}-run-${eval_run_seed}" \
        --policy=heuristic \
        --eval \
        --seed=${eval_run_seed} \
        --topology-init-seed=${topology_seed} \
        --train-topology-allow-eval-seed \
        --episode-steps=300 \
        --random-topology=0 \
        --n-data=20 \
        --n-router=20 \
        --degree=4 \
        --disable-progressbar \
        --eval-output-dir=${DIR_NAME}/eval_results_topo_${topology_seed}_run_${eval_run_seed}


done
)
