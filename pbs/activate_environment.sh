#!/bin/bash

# Load Conda, create the repository environment when necessary, and activate it.
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENVIRONMENT_NAME="${ENV_NAME:-thesis-testing}"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENVIRONMENT_NAME"; then
  echo "Creating Conda environment: $ENVIRONMENT_NAME"
  conda env create \
    --file environment.yml \
    --name "$ENVIRONMENT_NAME"
fi

conda activate "$ENVIRONMENT_NAME"
echo "Conda environment: $CONDA_DEFAULT_ENV"
