#!/bin/bash

# Output colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' 

# Configurable variables
HF_TOKEN="hf_your_actual_token_here"  # Replace with your Hugging Face token
MODEL_NAME="CraftJarvis/jarvis_vla_qwen2_vl_7b_sft"
LOCAL_MODEL_DIR="models/jarvis_vla_qwen2_vl_7b_sft"
RESOLUTION="1280x720"  # Reduced resolution to save memory
GPU_MEMORY_UTILIZATION=0.7  # Lower value for GPUs with limited VRAM
MAX_NUM_SEQS=2  # Reduced to conserve memory

# Error handling function
handle_error() {
    echo -e "${RED}Error in step: $1${NC}"
    echo -e "${YELLOW}Error message:${NC}"
    echo "$2"
    echo -e "${YELLOW}Attempting solution...${NC}"
    
    case $1 in
        "vllm_server")
            echo -e "${YELLOW}Retrying with reduced memory configuration...${NC}"
            start_vllm_server_with_retry
            ;;
        "simulator")
            echo -e "${YELLOW}Trying with software rendering...${NC}"
            xvfb-run -a python -m minestudio.simulator.entry
            ;;
        "model_download")
            echo -e "${YELLOW}Verifying Hugging Face connection and token...${NC}"
            check_hf_connection
            ;;
        *)
            echo -e "${RED}Could not resolve automatically.${NC}"
            exit 1
            ;;
    esac
}

# Verify Hugging Face connection
check_hf_connection() {
    if ! curl -s https://huggingface.co --connect-timeout 10 > /dev/null; then
        echo -e "${RED}Error: No connection to Hugging Face. Check your internet.${NC}"
        exit 1
    fi
    
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}Error: Hugging Face token not configured.${NC}"
        exit 1
    fi
    
    # Verify token validity
    if ! curl -s -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami > /dev/null; then
        echo -e "${RED}Error: Invalid Hugging Face token.${NC}"
        exit 1
    fi
}

# Start vllm server with retries
start_vllm_server_with_retry() {
    local retries=3
    local attempt=1
    
    while [ $attempt -le $retries ]; do
        echo -e "${YELLOW}Attempt $attempt to start vllm server...${NC}"
        
        # Clear GPU memory before each attempt
        nvidia-smi --gpu-reset -i 0 > /dev/null 2>&1
        
        # Start server with reduced configuration
        nohup bash -c "CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.api_server \
            --model $LOCAL_MODEL_DIR \
            --dtype float16 \
            --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
            --max-num-seqs $MAX_NUM_SEQS \
            --port 8000" > vllm_server.log 2>&1 &
        
        sleep 20  # Generous initialization time
        
        if curl -s http://localhost:8000/health > /dev/null; then
            echo -e "${GREEN}vllm server started successfully.${NC}"
            return 0
        else
            echo -e "${YELLOW}Server did not respond.${NC}"
            cat vllm_server.log | tail -n 20
            
            # Adjust parameters for next attempt
            GPU_MEMORY_UTILIZATION=$(echo "$GPU_MEMORY_UTILIZATION - 0.1" | bc)
            MAX_NUM_SEQS=$((MAX_NUM_SEQS - 1))
            
            # Ensure we don't go too low
            if (( $(echo "$GPU_MEMORY_UTILIZATION < 0.5" | bc -l) )); then
                GPU_MEMORY_UTILIZATION=0.5
            fi
            
            if [ $MAX_NUM_SEQS -lt 1 ]; then
                MAX_NUM_SEQS=1
            fi
            
            attempt=$((attempt + 1))
        fi
    done
    
    echo -e "${RED}Failed to start vllm server after $retries attempts.${NC}"
    exit 1
}

# Verify installations
verify_installation() {
    echo -e "\n${YELLOW}=== Verifying installations ===${NC}"
    
    # Check NVIDIA drivers
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: NVIDIA drivers not detected.${NC}"
        exit 1
    else
        echo -e "${GREEN}NVIDIA drivers detected.${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv
    fi
    
    # Check CUDA
    if ! nvcc --version &> /dev/null; then
        echo -e "${RED}Error: CUDA not detected.${NC}"
        exit 1
    else
        echo -e "${GREEN}CUDA detected.${NC}"
        nvcc --version
    fi
    
    # Check available VRAM
    local total_vram=$(nvidia-smi --query-gpu=memory.total --format=noheader,nounits | awk '{print $1}')
    if [ $total_vram -lt 4000 ]; then
        echo -e "${YELLOW}Warning: Only ${total_vram}MB VRAM available. Using reduced settings.${NC}"
        GPU_MEMORY_UTILIZATION=0.6
        MAX_NUM_SEQS=1
    fi
}

# Main installation
echo -e "${YELLOW}=== JARVIS-VLA Installation ===${NC}"

# 1. Update system
echo -e "\n${YELLOW}1. Updating system...${NC}"
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget git python3-pip python3-venv xvfb mesa-utils libgl1-mesa-glx

# 2. Install NVIDIA drivers (if missing)
echo -e "\n${YELLOW}2. Installing NVIDIA drivers...${NC}"
sudo ubuntu-drivers autoinstall

# 3. Install Miniconda
echo -e "\n${YELLOW}3. Installing Miniconda...${NC}"
if ! command -v conda &> /dev/null; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
else
    echo -e "${GREEN}Miniconda already installed.${NC}"
fi

# 4. Configure Conda environment
echo -e "\n${YELLOW}4. Setting up Conda environment...${NC}"
conda create -n jarvisvla python=3.10 -y || handle_error "conda_create" "Error creating conda environment"
conda activate jarvisvla

# 5. Install Java 8
echo -e "\n${YELLOW}5. Installing Java 8...${NC}"
conda install --channel=conda-forge openjdk=8 -y || handle_error "java_install" "Error installing Java"

# 6. Install PyTorch with CUDA support
echo -e "\n${YELLOW}6. Installing PyTorch with CUDA...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || handle_error "pytorch_install" "Error installing PyTorch"

# 7. Clone repository
echo -e "\n${YELLOW}7. Cloning JARVIS-VLA repository...${NC}"
if [ ! -d "JarvisVLA" ]; then
    git clone https://github.com/CraftJarvis/JarvisVLA.git || handle_error "git_clone" "Error cloning repository"
else
    echo -e "${GREEN}Repository exists. Updating...${NC}"
    cd JarvisVLA && git pull && cd ..
fi

# 8. Install dependencies
echo -e "\n${YELLOW}8. Installing dependencies...${NC}"
cd JarvisVLA
pip install -e . || handle_error "pip_install" "Error installing dependencies"
pip install vllm==0.3.3 huggingface-hub transformers==4.40.0 flash-attn --no-build-isolation || handle_error "pip_install_extra" "Error installing additional dependencies"

# 9. Configure memory management
echo -e "\n${YELLOW}9. Configuring memory settings...${NC}"
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,expandable_segments:True"' >> ~/.bashrc
echo 'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8' >> ~/.bashrc
source ~/.bashrc

# 10. Download model
echo -e "\n${YELLOW}10. Downloading JARVIS-VLA model...${NC}"
if [ ! -d "$LOCAL_MODEL_DIR" ]; then
    check_hf_connection
    huggingface-cli download $MODEL_NAME --token $HF_TOKEN --local-dir $LOCAL_MODEL_DIR || handle_error "model_download" "Error downloading model"
else
    echo -e "${GREEN}Model already downloaded.${NC}"
fi

# 11. Configure Minecraft resolution
echo -e "\n${YELLOW}11. Configuring Minecraft resolution...${NC}"
sed -i "s/SCREEN_WIDTH = .*/SCREEN_WIDTH = ${RESOLUTION%x*}/" minestudio/simulator/config.py || echo -e "${YELLOW}Warning: Could not adjust width${NC}"
sed -i "s/SCREEN_HEIGHT = .*/SCREEN_HEIGHT = ${RESOLUTION#*x}/" minestudio/simulator/config.py || echo -e "${YELLOW}Warning: Could not adjust height${NC}"

# Verify installations
verify_installation

# 12. Test simulator
echo -e "\n${YELLOW}12. Testing simulator...${NC}"
echo -e "${GREEN}Testing without GPU rendering:${NC}"
xvfb-run -a python -m minestudio.simulator.entry &
SIM_PID=$!
sleep 10
if ps -p $SIM_PID > /dev/null; then
    echo -e "${GREEN}Simulator working correctly.${NC}"
    kill $SIM_PID
else
    handle_error "simulator" "Simulator failed to start"
fi

# 13. Start inference server
echo -e "\n${YELLOW}13. Starting inference server...${NC}"
start_vllm_server_with_retry

# 14. Run Minecraft agent
echo -e "\n${YELLOW}14. Starting JARVIS-VLA in Minecraft...${NC}"
echo -e "${GREEN}Verifying server connection...${NC}"

# Wait for server to be ready
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}Server ready. Starting agent...${NC}"
        xvfb-run -a python scripts/evaluate/rollout-kill.py --base_url http://localhost:8000 || handle_error "agent_start" "Error starting agent"
        break
    else
        echo -e "${YELLOW}Waiting for server... (attempt $i/10)${NC}"
        sleep 10
    fi
done

if [ $i -eq 10 ]; then
    echo -e "${RED}Error: Could not connect to server after 10 attempts.${NC}"
    echo -e "${YELLOW}Server logs:${NC}"
    tail -n 50 vllm_server.log
    exit 1
fi

echo -e "\n${GREEN}=== Installation completed successfully! ===${NC}"
echo -e "To restart the system, run:"
echo -e "1. Inference server: ${YELLOW}cd JarvisVLA && conda activate jarvisvla && nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.api_server --model $LOCAL_MODEL_DIR --dtype float16 --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --max-num-seqs $MAX_NUM_SEQS --port 8000' > vllm_server.log 2>&1 &${NC}"
echo -e "2. Agent: ${YELLOW}cd JarvisVLA && conda activate jarvisvla && xvfb-run -a python scripts/evaluate/rollout-kill.py --base_url http://localhost:8000${NC}"
