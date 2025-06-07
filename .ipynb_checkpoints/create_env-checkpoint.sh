#!/bin/bash

# Define environment name and Python version
ENV_NAME="pose_estimation"
PYTHON_VERSION="3.11.9"  # Modify the Python version as needed
DEFORM_DETR_DIR="./Deformable-DETR"  
# Define the CUDA version (ensure you specify the version compatible with your system)
CUDA_VERSION="12.2"
# Define the CUDA version (ensure you specify the version compatible with your system)
CUDA_DRIVER_VERSION="12.0.0"


# Cloning Deformable DETR from github
echo "Cloning Deformable DETR from github"
git clone https://github.com/fundamentalvision/Deformable-DETR.git

# Create a new conda environment with the specified Python version
echo "Creating a Conda environment: $ENV_NAME with Python $PYTHON_VERSION"

if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Activating environment..."
    # Activate the environment
    source activate
    conda activate $ENV_NAME
else
    echo "Environment '$ENV_NAME' does not exist. Creating it..."
    # Create a new environment
    conda create -n $ENV_NAME python=$PYTHON_VERSION pip -y
    echo "Activating environment '$ENV_NAME'"
    # Activate the new environment
    source activate
    conda activate $ENV_NAME
fi

# Install torch and CUDA 
echo "Installing torch, torchvision using pip" # Using conda for installing torch - cuda not working 
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118  # cuda 12.1 not working for custon cuda operations with cuda 12 libs installed in system, changing pytorch to 11.8. 
pip install "numpy<2" # Error when torch is imported
pip install ninja # For parallel installation

#conda install pytorch=2.2.1 torchvision=0.17.1 cuda-toolkit=$CUDA_DRIVER_VERSION -c pytorch -c nvidia -y


# Install additional packages for deformable DETR
echo "Installing additional packages "
pip install -r $DEFORM_DETR_DIR"/requirements.txt"

# Verify the installation of CUDA
echo "Verifying CUDA installation..."
python -c "import torch; print(torch.cuda.is_available())"

cd $DEFORM_DETR_DIR"/models/ops"

sh ./make.sh

# unit test (should see all checking is True)
python test.py

# Return to the original directory
cd $OLDPWD

# To remove cache files
conda clean --all

