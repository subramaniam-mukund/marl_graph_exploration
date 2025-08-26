#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=4:mem=20gb
#PBS -l walltime=6:30:00
#PBS -o /rds/general/user/mss124/home/thesis/marl_graph_exploration/job_o
#PBS -e /rds/general/user/mss124/home/thesis/marl_graph_exploration/job_e

cd "$PBS_O_WORKDIR"

module purge
module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0
source ~/thesis/marl/bin/activate

echo "Running in: $(pwd)"

GPUS="0"

BASE_PARAMS="--gamma=0.9 --epsilon=1.0 --hidden-dim=512,256 --mini-batch-size=32 --device=cuda --lr=0.001 --tau=0.01 --step-before-train=10_000 --capacity=200_000 --eval-episodes=1000 --eval-episode-steps=300 --disable-progressbar"
LIMITS="--step-between-train=10 --total-steps=250_000"

DATE=$(date +%Y%m%d_%H%M%S)
DIR_NAME="${DATE}_logs_fixed_baseline"

mkdir -p $DIR_NAME

run() {
    RUN_NAME="$1"
    shift
    RUN_ARGS="$@"
    (set -x; time python -u src/main.py $RUN_ARGS --comment=${RUN_NAME}) > ${DIR_NAME}/${RUN_NAME}.log 2>&1
}

SELECTED_SEEDS=$(python -c "from src.env.constants_degree4 import SELECTED_SEEDS; print(*SELECTED_SEEDS)")

time (
# Graphs G_A, G_B, G_C, selected
for seed in 139608511
do
  for i in 0
  do
    for nh in 2 4 5
    do
      for loss_coeff in .1 1 5 10
      do
        run "fixed-dgn-t${seed}-intr${INTRINSIC_COEFF}-${i}" \
          --seed=$i \
          --topology-init-seed=$seed \
          --train-topology-allow-eval-seed \
          --episode-steps=300 \
          --model=dgn \
          --random-topology=0 \
          --num-heads=$nh \
          --intrinsic-coeff=1 \
          --intr-loss-coeff=$loss_coeff \
          --rnd-network=True --n-data=20 --n-router=20 --intr-reward-decay=.99 --enable-link-failures=True --link-failure-rate=.01 --degree=4 $BASE_PARAMS $LIMITS
      done
    done
  done
done
)