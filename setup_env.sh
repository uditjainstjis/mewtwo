#!/bin/bash
set -e

echo "🚀 Setting up Synapta v2.0 Environment..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check Python and venv
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 not found. Please install Python3.${NC}"
    exit 1
fi

VENV_DIR="/home/learner/Desktop/mewtwo/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment at $VENV_DIR...${NC}"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 2. Upgrade pip and install pip-tools
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# 3. Install requirements
echo -e "${GREEN}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt

# 4. Verify CUDA
echo -e "${GREEN}Verifying CUDA availability...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
cuda_avail = torch.cuda.is_available()
print(f'CUDA available: {cuda_avail}')
if cuda_avail:
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB')
else:
    print('WARNING: CUDA not detected! You will not be able to use the RTX 5090.')
"

echo -e "${GREEN}✅ Setup complete! run 'source .venv/bin/activate' to start.${NC}"
