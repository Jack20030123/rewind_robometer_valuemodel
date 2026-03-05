#!/usr/bin/env bash
set -euo pipefail

# Avoid nounset crash on some clusters
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate /project2/biyik_1165/haobaizh/rewind_robometer_valuemodel/conda_envs/rewind

export CC="$(which gcc)"
export CXX="$(which g++)"
export MUJOCO_GL=egl

export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"

# Make GCC runtime libs win over conda's libstdc++
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
GCC_LIBGCC_DIR="$(dirname "$(gcc -print-file-name=libgcc_s.so.1)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${GCC_LIBGCC_DIR}:${LD_LIBRARY_PATH}"

# MuJoCo runtime
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}"

# NVIDIA runtime (optional)
if [ -d /usr/lib/nvidia ]; then
  export LD_LIBRARY_PATH="/usr/lib/nvidia:${LD_LIBRARY_PATH}"
fi
if [ -d /usr/lib64/nvidia ]; then
  export LD_LIBRARY_PATH="/usr/lib64/nvidia:${LD_LIBRARY_PATH}"
fi
