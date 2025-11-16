#!/bin/bash

#  bash stop_training.sh [PID1] [PID2] ...

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "MedGMAE Graceful Training Shutdown Script"
echo "=========================================="


EXTRA_PIDS="$@"
if [ -n "$EXTRA_PIDS" ]; then
    echo -e "${YELLOW}Additional PIDs to stop: $EXTRA_PIDS${NC}"
fi

# PID文件
PID_FILE="train.pid"


TRAINING_SCRIPTS=(
    "MedGMAE/Gmain.py"
    "MedGMAE/Tmain.py"
)


find_training_processes() {
    local pattern=$(IFS='|'; echo "${TRAINING_SCRIPTS[*]}")
    local pids=$(pgrep -f "$pattern" 2>/dev/null)


    local torchrun_pids=$(ps aux | grep -E "torchrun.*MedGMAE|torchrun.*12333|torchrun.*Gmain.py" | grep -v grep | awk '{print $2}')


    echo "$pids $torchrun_pids" | tr ' ' '\n' | sort -u | tr '\n' ' '
}


show_process_info() {
    local pids=$1
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}Found training processes:${NC}"
        ps aux | head -1
        for pid in $pids; do
            ps aux | grep "^[^ ]* *$pid " | grep -v grep
        done
    else
        echo -e "${GREEN}✓ No training processes found${NC}"
    fi
}

check_gpu_status() {
    echo ""
    echo -e "${YELLOW}Checking GPU status (GPU 5,6,7)...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --id=5,6,7 --query-compute-apps=pid,process_name,used_memory --format=csv

        GPU_PIDS=$(nvidia-smi --id=5,6,7 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        if [ -n "$GPU_PIDS" ]; then
            echo -e "${YELLOW}⚠️  GPU processes still running (PIDs: $GPU_PIDS)${NC}"
            return 1
        else
            echo -e "${GREEN}✓ All training GPUs (5,6,7) are free${NC}"
            return 0
        fi
    else
        echo "nvidia-smi not available"
        return 0
    fi
}


force_cleanup_gpu() {
    echo ""
    echo -e "${YELLOW}Force cleaning GPU processes (GPU 5,6,7)...${NC}"

    if command -v nvidia-smi &> /dev/null; then
        GPU_PIDS=$(nvidia-smi --id=5,6,7 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        if [ -n "$GPU_PIDS" ]; then
            for pid in $GPU_PIDS; do

                if ps -p $pid -o comm= | grep -q python; then
                    echo "Killing GPU process: $pid"
                    kill -9 $pid 2>/dev/null
                fi
            done
            sleep 2
            check_gpu_status
        fi
    fi
}


MAIN_PID=""
if [ -f "$PID_FILE" ]; then
    MAIN_PID=$(cat $PID_FILE)
    echo "Main process PID from file: $MAIN_PID"


    if ! ps -p $MAIN_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Process $MAIN_PID does not exist${NC}"
        MAIN_PID=""
    fi
else
    echo -e "${YELLOW}⚠️  PID file not found: $PID_FILE${NC}"
fi


echo ""
echo "Searching for all MedGMAE training processes..."
ALL_PIDS=$(find_training_processes)

if [ -z "$ALL_PIDS" ] && [ -z "$MAIN_PID" ]; then
    echo -e "${GREEN}✓ No training processes found${NC}"
    check_gpu_status
    rm -f $PID_FILE
    exit 0
fi


if [ -n "$MAIN_PID" ]; then
    ALL_PIDS=$(echo "$MAIN_PID $ALL_PIDS" | tr ' ' '\n' | sort -u | tr '\n' ' ')
fi


if [ -n "$EXTRA_PIDS" ]; then
    ALL_PIDS=$(echo "$ALL_PIDS $EXTRA_PIDS" | tr ' ' '\n' | sort -u | tr '\n' ' ')
fi

echo ""
show_process_info "$ALL_PIDS"


echo ""
echo -e "${YELLOW}This will terminate all training processes.${NC}"
read -p "Continue? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi


echo ""
echo -e "${YELLOW}Sending SIGTERM for graceful shutdown...${NC}"
for pid in $ALL_PIDS; do
    kill -TERM $pid 2>/dev/null && echo "  Sent SIGTERM to PID $pid"
done


echo "Waiting for processes to terminate gracefully (max 30s)..."
WAIT_TIME=0
MAX_WAIT=30

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    REMAINING=$(find_training_processes)
    if [ -z "$REMAINING" ]; then
        echo -e "\n${GREEN}✓ All processes terminated gracefully${NC}"
        break
    fi
    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
    echo -n "."
done


REMAINING=$(find_training_processes)
if [ -n "$REMAINING" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Some processes did not stop gracefully${NC}"
    echo -e "${RED}Force killing remaining processes...${NC}"
    for pid in $REMAINING; do
        kill -9 $pid 2>/dev/null && echo "  Force killed PID $pid"
    done
    sleep 1
fi


STILL_RUNNING=$(find_training_processes)
if [ -n "$STILL_RUNNING" ]; then
    echo -e "${RED}❌ Error: Some processes still running: $STILL_RUNNING${NC}"
    show_process_info "$STILL_RUNNING"
else
    echo -e "${GREEN}✓ All training processes stopped${NC}"
fi


if ! check_gpu_status; then
    read -p "Force clean GPU processes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        force_cleanup_gpu
    fi
fi


if [ -f "$PID_FILE" ]; then
    rm -f $PID_FILE
    echo -e "${GREEN}✓ Removed PID file${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Shutdown complete!${NC}"
echo "=========================================="
echo ""
echo "To check logs:"
echo "  tail -f log_rebuttal_075_512.txt"
echo ""
echo "To resume training:"
echo "  CUDA_VISIBLE_DEVICES=5,6,7 nohup torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12333 Gmain.py > log_rebuttal_075_512.txt &"
echo ""
