#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=80gb
#PBS -l walltime=30:00:00
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
RECURRENT="--sequence-length=8"

DATE=$(date +%Y%m%d_%H%M%S)
DIR_NAME="baselines_20agents"

mkdir -p $DIR_NAME

run() {
    RUN_NAME="$1"
    shift
    RUN_ARGS="$@"
    (set -x; time python -u src/main.py $RUN_ARGS --comment=${RUN_NAME}) > ${DIR_NAME}/${RUN_NAME}.log 2>&1
}

SELECTED_SEEDS=$(python -c "from src.env.constants_degree4 import SELECTED_SEEDS; print(*SELECTED_SEEDS)")

time (
for seed in $SELECTED_SEEDS
do
for i in 0
do
run "fixed-shortest-paths-eval-t$seed-$i" --policy=heuristic --eval --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --random-topology=0 --disable-progressbar --n-data=20 --n-router=20 --enable-link-failures=True --link-failure-rate=.01 --degree=4 --eval-output-dir=${DIR_NAME}/fixed-shortest-paths-eval-t$seed-$i/eval
run "fixed-dqn-t$seed-$i" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=dqn --random-topology=0 --n-data=20 --n-router=20 --enable-link-failures=True --link-failure-rate=.01 --degree=4 $BASE_PARAMS $LIMITS
run "fixed-dqnr-t$seed-$i" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=dqnr --random-topology=0 --n-data=20 --n-router=20 --enable-link-failures=True --link-failure-rate=.01 --degree=4 $BASE_PARAMS $RECURRENT $LIMITS
run "fixed-tarmac-t${seed}-${i}" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=tarmac --random-topology=0 --n-data=20 --n-router=20 --enable-link-failures=True --link-failure-rate=.01 --degree=4 $BASE_PARAMS $RECURRENT $LIMITS
run "fixed-dgn-t$seed-$i" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=dgn --random-topology=0 --n-data=20 --n-router=20 --enable-link-failures=True --link-failure-rate=.01 --degree=4 $BASE_PARAMS $LIMITS
run "fixed-stdgn-t${seed}-${i}" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=stdgn --random-topology=0 --num-heads=8 --n-data=20 --n-router=20 --enable-link-failures=True --link-failure-rate=.01 --degree=4 $BASE_PARAMS $RECURRENT $LIMITS
run "fixed-stdgn+smce-t${seed}-${i}" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=stdgn --random-topology=0 --num-heads=8 --intrinsic-coeff=.1 --intr-loss-coeff=.1 --rnd-network=True --intr-reward-decay=.99 --n-data=20 --n-router=20 --enable-link-failures=True --link-failure-rate=.01 --degree=4 $BASE_PARAMS $RECURRENT $LIMITS
done
done
)
