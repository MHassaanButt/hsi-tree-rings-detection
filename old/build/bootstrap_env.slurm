#!/bin/bash
#SBATCH --job-name=bootstrap-env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=//mnt/users/mubut0522/repos/tree_wood_specie_classification/logs/bootstrap_%j.out

module load singularity

SIF=/mnt/users/mubut0522/containers/pytorch_2.1.1_cu121.sif
PROJECT=/mnt/users/mubut0522/repos/tree_wood_specie_classification/
REQS=$PROJECT/build/requirements.txt   # adjust if your requirements.txt lives elsewhere

# Make sure ~/.local/bin is on PATH so 'jupyter' is found
export PATH="$HOME/.local/bin:$PATH"

echo "Node: $(hostname)"
nvidia-smi || true

# Install everything into your HOME (persistent), not inside the SIF
singularity exec --nv -B $PROJECT:/workspace $SIF bash -lc "
  python -V
  python -m pip install --user --upgrade pip setuptools wheel
  python -m pip install --user 'tensorflow[and-cuda]==2.15.1'
  python -m pip install --user torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --extra-index-url https://download.pytorch.org/whl/cu121
  python -m pip install --user notebook ipykernel
  [ -f $REQS ] && python -m pip install --user -r $REQS || true

  echo '--- Sanity checks ---'
  python - <<'PY'
import torch, tensorflow as tf
print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
print('TF:', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))
PY
"
