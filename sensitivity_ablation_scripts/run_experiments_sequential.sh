#!/bin/bash
# Sequential Training Script for StaR Models
# This script runs multiple training commands one after another with individual logging

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create logs directory if it doesn't exist
LOGS_DIR="experiment_logs"
mkdir -p "$LOGS_DIR"
echo "Created/verified directory: $LOGS_DIR"

# Get timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_LOG="$LOGS_DIR/summary_$TIMESTAMP.log"

# Initialize summary log
{
    echo "================================================================================"
    echo "StaR Sequential Training Summary"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    echo ""
} > "$SUMMARY_LOG"

# Define commands to run in order
declare -a NAMES=(
    "linear"
    "random_connection_0.45"
    "nonlinear"
    "random_connection_0.85"
    "random_connection_0.55"
    "random_connection_0.75"
)

declare -a COMMANDS=(
    "uv run python main_tgn_original.py --dataset_name linear"
    "uv run python main_tgn_original.py --dataset_name random_connection --base_connection_prob 0.45 --preprocessing_data 1"
    "uv run python main_tgn_original.py --dataset_name nonlinear"
    "uv run python main_tgn_original.py --dataset_name random_connection --base_connection_prob 0.85 --preprocessing_data 1"
    "uv run python main_tgn_original.py --dataset_name random_connection --base_connection_prob 0.55 --preprocessing_data 1"
    "uv run python main_tgn_original.py --dataset_name random_connection --base_connection_prob 0.75 --preprocessing_data 1"
)

declare -a LOG_FILES=(
    "$LOGS_DIR/exp1_linear_$TIMESTAMP.log"
    "$LOGS_DIR/exp2_random_0.45_$TIMESTAMP.log"
    "$LOGS_DIR/exp3_nonlinear_$TIMESTAMP.log"
    "$LOGS_DIR/exp4_random_0.85_$TIMESTAMP.log"
    "$LOGS_DIR/exp5_random_0.55_$TIMESTAMP.log"
    "$LOGS_DIR/exp6_random_0.75_$TIMESTAMP.log"
)

# Track results
TOTAL_COMMANDS=${#COMMANDS[@]}
SUCCESS_COUNT=0
declare -a FAILED_COMMANDS=()

# Run each command sequentially
for i in "${!COMMANDS[@]}"; do
    CMD_NUM=$((i + 1))
    NAME="${NAMES[$i]}"
    CMD="${COMMANDS[$i]}"
    LOG_FILE="${LOG_FILES[$i]}"
    
    echo -e "\n${CYAN}================================================================================${NC}"
    echo -e "${CYAN}[$CMD_NUM/$TOTAL_COMMANDS] Running: $NAME${NC}"
    echo -e "${CYAN}================================================================================${NC}"
    echo -e "${YELLOW}Command: $CMD${NC}"
    echo -e "${YELLOW}Log file: $LOG_FILE${NC}"
    echo -e "${GREEN}Started at: $(date '+%Y-%m-%d %H:%M:%S')${NC}\n"
    
    # Log to summary
    {
        echo ""
        echo "[$CMD_NUM/$TOTAL_COMMANDS] Running: $NAME"
        echo "Command: $CMD"
        echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    } >> "$SUMMARY_LOG"
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Run command and capture output to log file
    if eval "$CMD" 2>&1 | tee "$LOG_FILE"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        DURATION_FMT=$(printf '%02d:%02d:%02d' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))
        
        echo -e "\n${GREEN}================================================================================${NC}"
        echo -e "${GREEN}SUCCESS: $NAME completed${NC}"
        echo -e "${GREEN}Duration: $DURATION_FMT${NC}"
        echo -e "${GREEN}================================================================================${NC}\n"
        
        {
            echo "Status: SUCCESS"
            echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Duration: $DURATION_FMT"
        } >> "$SUMMARY_LOG"
        
        ((SUCCESS_COUNT++))
    else
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        DURATION_FMT=$(printf '%02d:%02d:%02d' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))
        
        echo -e "\n${RED}================================================================================${NC}"
        echo -e "${RED}ERROR: $NAME failed with exit code $EXIT_CODE${NC}"
        echo -e "${RED}Duration: $DURATION_FMT${NC}"
        echo -e "${RED}Check log file for details: $LOG_FILE${NC}"
        echo -e "${RED}================================================================================${NC}\n"
        
        {
            echo "Status: FAILED (Exit code: $EXIT_CODE)"
            echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Duration: $DURATION_FMT"
        } >> "$SUMMARY_LOG"
        
        FAILED_COMMANDS+=("$NAME")
    fi
done

# Final summary
echo -e "\n\n${CYAN}================================================================================${NC}"
echo -e "${CYAN}ALL EXPERIMENTS COMPLETED${NC}"
echo -e "${CYAN}================================================================================${NC}"

{
    echo ""
    echo ""
    echo "================================================================================"
    echo "FINAL SUMMARY"
    echo "================================================================================"
    echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Total commands: $TOTAL_COMMANDS"
    echo "Successful: $SUCCESS_COUNT"
    echo "Failed: ${#FAILED_COMMANDS[@]}"
} >> "$SUMMARY_LOG"

echo "Total commands: $TOTAL_COMMANDS"
echo -e "Successful: ${GREEN}$SUCCESS_COUNT${NC}"
if [ ${#FAILED_COMMANDS[@]} -gt 0 ]; then
    echo -e "Failed: ${RED}${#FAILED_COMMANDS[@]}${NC}"
else
    echo -e "Failed: ${GREEN}${#FAILED_COMMANDS[@]}${NC}"
fi

if [ ${#FAILED_COMMANDS[@]} -gt 0 ]; then
    echo -e "\n${RED}Failed experiments:${NC}"
    echo "" >> "$SUMMARY_LOG"
    echo "Failed experiments:" >> "$SUMMARY_LOG"
    for failed in "${FAILED_COMMANDS[@]}"; do
        echo -e "${RED}  - $failed${NC}"
        echo "  - $failed" >> "$SUMMARY_LOG"
    done
fi

echo -e "\n${YELLOW}All logs saved in: $LOGS_DIR${NC}"
echo -e "${YELLOW}Summary log: $SUMMARY_LOG${NC}"
echo -e "${CYAN}================================================================================${NC}\n"

echo "================================================================================" >> "$SUMMARY_LOG"

# Exit with appropriate code
if [ ${#FAILED_COMMANDS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
