#!/bin/bash
# find_training_pids.sh 

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "MedGMAE Training Process Finder"
echo "==========================================${NC}"
echo ""


echo -e "${YELLOW}Searching for training processes...${NC}"
echo ""


GMAIN_PIDS=$(pgrep -f "Gmain.py" 2>/dev/null)
if [ -n "$GMAIN_PIDS" ]; then
    echo -e "${GREEN}✓ Found Gmain.py processes:${NC}"
    ps aux | head -1
    for pid in $GMAIN_PIDS; do
        ps aux | grep "^[^ ]* *$pid " | grep -v grep
    done
    echo ""
fi


TORCHRUN_PIDS=$(ps aux | grep -E "torchrun.*MedGMAE|torchrun.*12333" | grep -v grep | awk '{print $2}')
if [ -n "$TORCHRUN_PIDS" ]; then
    echo -e "${GREEN}✓ Found torchrun processes (MedGMAE/port 12333):${NC}"
    ps aux | head -1
    for pid in $TORCHRUN_PIDS; do
        ps aux | grep "^[^ ]* *$pid " | grep -v grep
    done
    echo ""
fi


NOHUP_PIDS=$(ps aux | grep nohup | grep -E "Gmain.py|Tmain.py" | grep -v grep | awk '{print $2}')
if [ -n "$NOHUP_PIDS" ]; then
    echo -e "${GREEN}✓ Found nohup processes:${NC}"
    ps aux | head -1
    for pid in $NOHUP_PIDS; do
        ps aux | grep "^[^ ]* *$pid " | grep -v grep
    done
    echo ""
fi


PYTHON_PIDS=$(ps aux | grep python | grep -E "Gmain.py|Tmain.py" | grep -v grep | awk '{print $2}')
if [ -n "$PYTHON_PIDS" ]; then
    echo -e "${GREEN}✓ Found Python processes:${NC}"
    ps aux | head -1
    for pid in $PYTHON_PIDS; do
        ps aux | grep "^[^ ]* *$pid " | grep -v grep
    done
    echo ""
fi

ALL_PIDS=$(echo "$GMAIN_PIDS $TORCHRUN_PIDS $NOHUP_PIDS $PYTHON_PIDS" | tr ' ' '\n' | sort -u | tr '\n' ' ')

echo ""
echo -e "${BLUE}==========================================${NC}"
if [ -n "$ALL_PIDS" ]; then
    echo -e "${YELLOW}Summary - All unique PIDs:${NC}"
    echo -e "${GREEN}$ALL_PIDS${NC}"
    echo ""
    echo -e "${YELLOW}To stop these processes, use:${NC}"
    echo -e "  ${GREEN}bash stop_now.sh $ALL_PIDS${NC}"
    echo -e "  ${GREEN}bash stop_training.sh $ALL_PIDS${NC}"
else
    echo -e "${GREEN}✓ No training processes found${NC}"
fi
echo -e "${BLUE}==========================================${NC}"
echo ""


if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}GPU Status (GPU 5,6,7):${NC}"
    nvidia-smi --id=5,6,7 --query-compute-apps=pid,process_name,used_memory --format=csv
    echo ""
fi
