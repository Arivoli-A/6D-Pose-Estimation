#!/bin/bash

# --- Script to Set Up Conda Activate Environment Variables ---
# This script creates a file in your active conda environment's
# 'etc/conda/activate.d/' directory to automatically set
# LD_LIBRARY_PATH for PyTorch and libcudart.so.12.

echo "--- Setting up Conda Environment Variables for PyTorch and CUDA ---"

# 1. Check if a conda environment is active
if [ -z "$CONDA_PREFIX" ]; then
  echo "Error: No conda environment is currently active."
  echo "Please activate your target conda environment first (e.g., 'conda activate myenv') and then run this script."
  exit 1
fi

echo "Currently active conda environment: $(basename "$CONDA_PREFIX")"
echo "Conda environment prefix: $CONDA_PREFIX"

read -p "Do you want to set up the LD_LIBRARY_PATH script in this environment? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Setup cancelled by user."
  exit 0
fi

# 2. Define the path for the activate.d directory and the script file
ACTIVATE_D_DIR="$CONDA_PREFIX/etc/conda/activate.d"
SCRIPT_FILE="$ACTIVATE_D_DIR/env_vars.sh"

# 3. Create the activate.d directory if it doesn't exist
echo "Creating directory: $ACTIVATE_D_DIR (if it doesn't exist)"
mkdir -p "$ACTIVATE_D_DIR"

# 4. Write the content to the env_vars.sh script
echo "Creating/Updating script file: $SCRIPT_FILE"
cat << 'EOF' > "$SCRIPT_FILE"
#!/bin/bash

# This script is sourced by conda activate to set LD_LIBRARY_PATH.
# It identifies PyTorch's lib path and the libcudart.so.12 directory
# within the active conda environment and prepends them to LD_LIBRARY_PATH
# if they are not already present.

# Get PyTorch's library path
TORCH_LIB_PATH=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$TORCH_LIB_PATH" ]; then
    # Fallback if python -c fails or returns empty, e.g., if torch is not installed
    TORCH_LIB_PATH=""
    echo "Warning: PyTorch not found or could not determine its lib path."
fi


# Find the directory containing libcudart.so.12
LIBCUDART_DIR=$(find "$CONDA_PREFIX" -name "libcudart.so.12" 2>/dev/null | xargs dirname 2>/dev/null)
if [ -z "$LIBCUDART_DIR" ]; then
    echo "Warning: libcudart.so.12 not found within CONDA_PREFIX. Ensure CUDA toolkit is installed if needed."
fi

# Build a list of paths to add, ensuring no duplicates and prepending if already present
NEW_PATHS=""

# Add PyTorch's lib path if it's found and not already in LD_LIBRARY_PATH
if [ -n "$TORCH_LIB_PATH" ] && [[ ":$LD_LIBRARY_PATH:" != *":$TORCH_LIB_PATH:"* ]]; then
  NEW_PATHS="$TORCH_LIB_PATH"
  echo "Preparing to add PyTorch lib path: $TORCH_LIB_PATH"
fi

# Add libcudart directory if it's found and not already in LD_LIBRARY_PATH
# Prepend it to NEW_PATHS to give it higher priority if both are added
if [ -n "$LIBCUDART_DIR" ] && [[ ":$LD_LIBRARY_PATH:" != *":$LIBCUDART_DIR:"* ]]; then
  if [ -n "$NEW_PATHS" ]; then
    NEW_PATHS="$LIBCUDART_DIR:$NEW_PATHS"
  else
    NEW_PATHS="$LIBCUDART_DIR"
  fi
  echo "Preparing to add libcudart directory: $LIBCUDART_DIR"
fi

# Export LD_LIBRARY_PATH if there are new paths to add
if [ -n "$NEW_PATHS" ]; then
  if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$NEW_PATHS:$LD_LIBRARY_PATH"
    echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="$NEW_PATHS"
    echo "Set LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
  fi
else
  echo "No new paths added to LD_LIBRARY_PATH for PyTorch/CUDA."
fi
EOF

# Make the script executable (though sourcing doesn't strictly require this, it's good practice)
chmod +x "$SCRIPT_FILE"

echo "--- Setup Complete! ---"
echo ""
echo "To test the changes:"
echo "1. Deactivate your current conda environment:"
echo "   conda deactivate"
echo ""
echo "2. Reactivate the environment you just configured ($(basename "$CONDA_PREFIX")):"
echo "   conda activate $(basename "$CONDA_PREFIX")"
echo ""
echo "3. After reactivation, check the LD_LIBRARY_PATH:"
echo "   echo \$LD_LIBRARY_PATH"
echo ""
echo "You should see the PyTorch and libcudart paths at the beginning of LD_LIBRARY_PATH."