#!/bin/bash

# Script to run the LoRA fine-tuning pipeline
# Usage: ./run_sft.sh [test|train] [arguments]
#
# Examples:
#   ./run_sft.sh test --gpu_ids 5
#   ./run_sft.sh train --gpu_ids 7 --output_dir output/model

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"  # Change to script directory

# Check if required Python packages are installed
check_requirements() {
    echo "Checking requirements..."
    
    # Map package names to their import names (when different)
    declare -A PACKAGE_MAP
    PACKAGE_MAP["transformers"]="transformers"
    PACKAGE_MAP["peft"]="peft"
    PACKAGE_MAP["accelerate"]="accelerate"
    PACKAGE_MAP["bitsandbytes"]="bitsandbytes"
    PACKAGE_MAP["torch"]="torch"
    PACKAGE_MAP["datasets"]="datasets"
    PACKAGE_MAP["tqdm"]="tqdm"
    PACKAGE_MAP["python-dotenv"]="dotenv"
    
    MISSING_PACKAGES=()
    
    for package in "${!PACKAGE_MAP[@]}"; do
        import_name="${PACKAGE_MAP[$package]}"
        # Use the same Python interpreter that will run the script
        if ! python -c "import $import_name" &> /dev/null; then
            MISSING_PACKAGES+=("$package")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo "The following packages appear to be missing:"
        for pkg in "${MISSING_PACKAGES[@]}"; do
            echo "  - $pkg"
        done
        echo "Please install the required packages with:"
        echo "pip install ${MISSING_PACKAGES[*]}"
        
        # Ask if user wants to continue anyway
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "All required packages are installed."
    fi
}

# Check GPU availability
check_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "⚠️ nvidia-smi command not found. Are NVIDIA drivers installed?"
        return
    fi
    
    echo "Checking available GPUs..."
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
    
    # If GPU ID specified, check if it exists
    if [ -n "$GPU_IDS" ]; then
        # Extract first GPU ID if multiple are provided
        FIRST_GPU=$(echo "$GPU_IDS" | cut -d ',' -f1)
        echo "Note: Only using GPU $FIRST_GPU (for simplicity)"
        
        if ! nvidia-smi -i "$FIRST_GPU" &> /dev/null; then
            echo "⚠️ Warning: GPU $FIRST_GPU does not appear to exist or is not accessible."
        fi
    fi
}

# Run the test pipeline for verification
run_test() {
    echo "Running SFT pipeline test..."
    
    GPU_ARGS=""
    if [ -n "$GPU_IDS" ]; then
        GPU_ARGS="--gpu_ids $GPU_IDS"
        FIRST_GPU=$(echo "$GPU_IDS" | cut -d ',' -f1)
        echo "Using GPU: $FIRST_GPU"
    fi
    
    python test_sft.py $GPU_ARGS
    status=$?
    if [ $status -eq 0 ]; then
        echo "✅ Test completed successfully!"
    else
        echo "❌ Test failed. Fix issues before proceeding to actual training."
        exit $status
    fi
}

# Run the full fine-tuning pipeline
run_sft() {
    echo "Running full SFT pipeline..."
    
    # Default values
    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    DATA_PATH="test_data.json"
    OUTPUT_DIR="output/sft-model"
    BATCH_SIZE=4
    LEARNING_RATE="2e-4"
    NUM_EPOCHS=3
    MAX_LENGTH=1024
    GRAD_ACCUM=8
    LORA_R=16
    LORA_ALPHA=32
    LORA_DROPOUT=0.05
    GPU_IDS=""
    
    # Parse custom command line args
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model_name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --traindata_path)
                DATA_PATH="$2"
                shift 2
                ;;
            --output_dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --batch_size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --learning_rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --num_epochs)
                NUM_EPOCHS="$2"
                shift 2
                ;;
            --max_length)
                MAX_LENGTH="$2"
                shift 2
                ;;
            --gradient_accumulation_steps)
                GRAD_ACCUM="$2"
                shift 2
                ;;
            --lora_r)
                LORA_R="$2"
                shift 2
                ;;
            --lora_alpha)
                LORA_ALPHA="$2"
                shift 2
                ;;
            --lora_dropout)
                LORA_DROPOUT="$2"
                shift 2
                ;;
            --gpu_ids)
                GPU_IDS="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                exit 1
                ;;
        esac
    done
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Add GPU argument if specified
    GPU_ARGS=""
    if [ -n "$GPU_IDS" ]; then
        GPU_ARGS="--gpu_ids $GPU_IDS"
        FIRST_GPU=$(echo "$GPU_IDS" | cut -d ',' -f1)
        echo "Using GPU: $FIRST_GPU"
    fi
    
    # Run the actual training
    python sft.py \
        --model_name "$MODEL_NAME" \
        --traindata_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --num_epochs "$NUM_EPOCHS" \
        --max_length "$MAX_LENGTH" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        $GPU_ARGS
        
    status=$?
    if [ $status -eq 0 ]; then
        echo "✅ SFT completed successfully!"
        echo "Model saved to: $OUTPUT_DIR"
    else
        echo "❌ SFT failed with status code $status"
        exit $status
    fi
}

# Extract GPU IDs from arguments
extract_gpu_ids() {
    for ((i=1; i<=$#; i++)); do
        if [[ "${!i}" == "--gpu_ids" ]]; then
            next=$((i+1))
            if [[ $next -le $# ]]; then
                GPU_IDS="${!next}"
                return
            fi
        fi
    done
}

# Main execution
check_requirements

if [ $# -eq 0 ]; then
    echo "No command specified. Use 'test' or 'train'."
    echo "Usage: ./run_sft.sh [test|train] [arguments]"
    echo "Example: ./run_sft.sh train --gpu_ids 5 --output_dir output/model"
    exit 1
fi

command="$1"
shift  # Remove the command from args

# Extract GPU IDs if provided
extract_gpu_ids "$@"

# Check GPU availability
check_gpus

case "$command" in
    test)
        run_test
        ;;
    train)
        run_sft "$@"
        ;;
    *)
        echo "Unknown command: $command"
        echo "Usage: ./run_sft.sh [test|train] [arguments]"
        echo "Example: ./run_sft.sh train --gpu_ids 5 --output_dir output/model"
        exit 1
        ;;
esac

exit 0 