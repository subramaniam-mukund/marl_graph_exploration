#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:gpu_type=L40S:mem=48gb
#PBS -l walltime=05:00:00
#PBS -o /rds/general/user/mss124/home/thesis/job_o
#PBS -e /rds/general/user/mss124/home/thesis/job_e

cd "$PBS_O_WORKDIR"

module purge
module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0
source ~/thesis/marl/bin/activate
echo "Running in: $(pwd)"

CUDA_VISIBLE_DEVICES=0

BASE_PARAMS="--step-between-train=10 --total-steps=2_500_000 --netmon --model=dqn --random-topology=1 --gamma=0.9 --epsilon=1.0 --epsilon-decay=0.999 --hidden-dim=512,256 --netmon-encoder-dim=512,256 --netmon-dim=128 --mini-batch-size=32 --device=cuda --lr=0.001 --tau=0.01 --step-before-train=100_000 --capacity=200_000 --eval-episodes=1000 --eval-episode-steps=300 --episode-steps=50 --disable-progressbar"
DATE=$(date +%Y%m%d_%H%M%S)
DIR_NAME="${DATE}_logs_dynamic"
mkdir -p $DIR_NAME


run() {
    RUN_NAME="$1"
    shift
    RUN_ARGS="$@"
    (set -x; time python -u src/main.py $RUN_ARGS --comment=${RUN_NAME}) > ${DIR_NAME}/${RUN_NAME}.log 2>&1
}

time (
for i in 0
do
for nocong in 1
do
if [ "$nocong" -eq "0" ]; then
   NOCONG_RUNNAME=""
   NOCONG_ARG=""
else
   NOCONG_RUNNAME="-nocong"
   NOCONG_ARG="--no-congestion"
fi
run "dynamic${NOCONG_RUNNAME}-netmon-1it-8seq-$i" --netmon-agg-type=sum --netmon-rnn-type=lstm --netmon-iterations=1 --sequence-length=8 --seed=$i $NOCONG_ARG $BASE_PARAMS
run "dynamic${NOCONG_RUNNAME}-netmon-gconvlstm-1it-8seq-$i" --netmon-agg-type=gconvlstm --netmon-rnn-type=gconvlstm --netmon-iterations=1 --sequence-length=8 --seed=$i $NOCONG_ARG $BASE_PARAMS
run "dynamic${NOCONG_RUNNAME}-netmon-graphsage-8it-1seq-$i" --netmon-agg-type=graphsage --netmon-rnn-type=none --netmon-iterations=8 --sequence-length=1 --seed=$i $NOCONG_ARG $BASE_PARAMS
run "dynamic${NOCONG_RUNNAME}-netmon-antisymgcn-8it-1seq-$i" --netmon-agg-type=antisymgcn --netmon-rnn-type=none --netmon-iterations=8 --sequence-length=1 --seed=$i $NOCONG_ARG $BASE_PARAMS
run "dynamic${NOCONG_RUNNAME}-shortest-paths-eval-$i" --policy=heuristic --eval --random-topology=1 --disable-progressbar --eval-output-dir=${DIR_NAME}/dynamic${NOCONG_RUNNAME}-shortest-paths-eval-$i/eval --seed=$i $NOCONG_ARG
done
done
)
