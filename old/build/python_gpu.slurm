#!/bin/bash
#SBATCH --job-name=py-hsi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/users/mubut0522/repos/hsi-tree-rings-detection/logs/py_%j.out
set -euo pipefail

module load singularity

# --- Paths ---
SIF=/mnt/users/mubut0522/containers/pytorch_2.1.1_cu121.sif
PROJECT=/mnt/users/mubut0522/repos/hsi-tree-rings-detection
SCRIPT=$PROJECT/unmix_spa_fcls.py
ARGS="  --cube processed_cubes/ipe_block0_wl.npz \
  --k 3 \
  --exclude-wl 1350 1450 1800 1950 2450 2538 \
  --mnf 0 \
  --tv 0.15 \
  --out unmixing_outputs/unmix_ipe_block0_wl_k3_fixed
"

# Make your user-installed tools visible INSIDE the container (prepend; don’t replace PATH)
export APPTAINERENV_PREPEND_PATH="$HOME/.local/bin"

# Ensure user site-packages are importable; handle unset PYTHONPATH safely
export APPTAINERENV_PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"

echo "Node: $(hostname -f)"
nvidia-smi || true

# Run your script with the container’s Python
singularity exec --nv -B "$PROJECT":/workspace "$SIF" \
  /opt/conda/bin/python /workspace/$(basename "$SCRIPT") $ARGS

# Optional GPU checks
# singularity exec --nv "$SIF" /opt/conda/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# singularity exec --nv "$SIF" /opt/conda/bin/python -c "import tensorflow as tf; print('tf', tf.__version__, 'gpus', tf.config.list_physical_devices('GPU'))"
echo "========================================Job completed successfully.================================"