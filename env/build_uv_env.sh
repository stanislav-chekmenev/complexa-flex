#!/bin/bash
set -e

# ==============================================================================
# Protein Foundation Models - Public UV Environment Installation Script
# ==============================================================================
# Public release variant: installs all Python dependencies (--full) by default.
# Does NOT install Foldseek, MMseqs2, Foldcomp, or any NVIDIA-internal binaries.
#
# Usage: ./build_public_uv_env.sh [OPTIONS] [INSTALL_ROOT]
#
# Options:
#   --clean         Remove existing .venv and UV cache before building (fresh start)
#   --minimal       Skip optional dependencies (JAX, ColabFold, tmol)
#   --python VER    Python version: 3.11 or 3.12 (default: 3.12)
#   --name NAME     Custom prompt name shown when venv is activated (default: complexa)
#   --root PATH     Specify installation root directory (where .venv will be created)
#   -h, --help      Show this help message
#
# Examples:
#   ./build_public_uv_env.sh                      # Full install (Python 3.12)
#   ./build_public_uv_env.sh --minimal            # Base dependencies only
#   ./build_public_uv_env.sh --python 3.11        # Full install with Python 3.11
#   ./build_public_uv_env.sh --name myenv         # Custom prompt: (myenv)
#   ./build_public_uv_env.sh --root /path/to/dir  # Create .venv in custom directory
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INSTALL_ROOT=""
FULL_INSTALL=true
CLEAN=false
PYTHON_VERSION="3.12"
VENV_NAME="complexa"

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --minimal)
            FULL_INSTALL=false
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            if [[ "$PYTHON_VERSION" != "3.11" && "$PYTHON_VERSION" != "3.12" ]]; then
                echo "Error: Python version must be 3.11 or 3.12"
                exit 1
            fi
            shift 2
            ;;
        --name)
            VENV_NAME="$2"
            shift 2
            ;;
        --root)
            INSTALL_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./build_public_uv_env.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean         Remove existing .venv and UV cache before building"
            echo "  --minimal       Skip optional dependencies (JAX, ColabFold, tmol)"
            echo "  --python VER    Python version: 3.11 or 3.12 (default: 3.12)"
            echo "  --name NAME     Custom prompt name (default: complexa)"
            echo "  --root PATH     Specify where to create .venv (default: project dir)"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./build_public_uv_env.sh                # Full install (Python 3.12)"
            echo "  ./build_public_uv_env.sh --minimal      # Base dependencies only"
            echo "  ./build_public_uv_env.sh --python 3.11  # Full install with Python 3.11"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$INSTALL_ROOT" ]]; then
                INSTALL_ROOT="$1"
            else
                echo "Error: Multiple paths specified"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -n "$INSTALL_ROOT" ]]; then
    VENV_DIR="$INSTALL_ROOT"
    mkdir -p "$VENV_DIR"
else
    VENV_DIR="$PROJECT_DIR"
fi

cd "$VENV_DIR"

export UV_CACHE_DIR="$VENV_DIR/.uv-cache"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

if [[ "$CLEAN" == "true" ]]; then
    echo "Cleaning existing environment and cache..."
    rm -rf "$VENV_DIR/.venv"
    rm -rf "$UV_CACHE_DIR"
    uv cache clean 2>/dev/null || true
    echo "Clean complete."
fi

echo "=============================================="
echo "  Proteina-Complexa - Public Environment Builder"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Install directory: $VENV_DIR"
echo "Python version: $PYTHON_VERSION"
echo "Prompt name: $VENV_NAME"
echo "Full install: $FULL_INSTALL"
echo ""

# ------------------------------------------------------------------------------
# 1. Check/Install UV
# ------------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "[1/9] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/9] uv already installed: $(uv --version)"
fi

# ------------------------------------------------------------------------------
# 2. Create virtual environment
# ------------------------------------------------------------------------------
echo "[2/9] Creating virtual environment with Python $PYTHON_VERSION..."
uv venv --python "$PYTHON_VERSION" --prompt "$VENV_NAME" "$VENV_DIR/.venv"

source "$VENV_DIR/.venv/bin/activate"
echo "      Python: $(which python)"

# ------------------------------------------------------------------------------
# 3. Install PyTorch with CUDA 13.0
# ------------------------------------------------------------------------------
echo "[3/9] Installing PyTorch 2.10.0 with CUDA 13.0..."
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# ------------------------------------------------------------------------------
# 4. Install base dependencies from pyproject.toml
# ------------------------------------------------------------------------------
echo "[4/9] Installing base dependencies from pyproject.toml..."
uv pip install --index-strategy unsafe-best-match -e "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# 5. Install PyTorch Geometric packages
# ------------------------------------------------------------------------------
echo "[5/9] Installing PyTorch Geometric packages..."
uv pip install torch_geometric torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.10.0+cu130.html

# ------------------------------------------------------------------------------
# 6. Install Graphein and Atomworks
# ------------------------------------------------------------------------------
echo "[6/9] Installing Graphein and Atomworks..."
echo "      -> Graphein..."
uv pip install graphein==1.7.7 --no-deps

echo "      -> Atomworks..."
uv pip install --index-strategy unsafe-best-match "atomworks[ml,openbabel,dev]" || echo "Warning: atomworks install failed"

# ------------------------------------------------------------------------------
# 7. Install optional/full dependencies
# ------------------------------------------------------------------------------
if [ "$FULL_INSTALL" = true ]; then
    echo "[7/9] Installing full dependencies (ColabFold, JAX, tmol)..."

    echo "      -> ColabDesign & AlphaFold-ColabFold..."
    uv pip install colabdesign==1.1.1 alphafold-colabfold==2.3.7

    echo "      -> Installing local colabdesign (community_models/colabdesign)..."
    uv pip install -e "$PROJECT_DIR/community_models/colabdesign"

    echo "      -> JAX with CUDA 13..."
    uv pip install "jax[cuda13]==0.9.2"
    uv pip install flax==0.9.0 --no-deps

    echo "      -> Tmol..."
    uv pip install "git+https://github.com/uw-ipd/tmol.git@d8a6f7f9649d36e74440bca25246ee7c467ce490" || echo "Warning: tmol install failed"
else
    echo "[7/9] Skipping optional dependencies (omit --minimal to install)"
fi

# ------------------------------------------------------------------------------
# 8. Install Foundry (Python version dependent)
# ------------------------------------------------------------------------------
if [[ "$PYTHON_VERSION" == "3.12" ]]; then
    echo "[8/9] Installing Foundry (rc-foundry)..."
    uv pip install --index-strategy unsafe-best-match "rc-foundry[all]"
else
    echo "[8/9] Not able to install RF# via rc-foundary. Please try python 3.12."
fi

uv pip install biotite==1.6.0
echo "Updated biotite to 1.6.0 for ligand compatibility"

# ------------------------------------------------------------------------------
# 9. Re-pin PyTorch + Geometric to correct CUDA 13.0 versions
#    (atomworks and other packages may downgrade torch during install)
# ------------------------------------------------------------------------------
echo "[9/9] Re-pinning PyTorch 2.10.0 + CUDA 13.0 (fixing any downgrades)..."

echo "      -> Removing conflicting CUDA 12 packages..."
uv pip uninstall \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 nvidia-nvshmem-cu12 \
    cuequivariance-ops-cu12 cuequivariance-ops-torch-cu12 \
    2>/dev/null || true

echo "      -> Reinstalling PyTorch 2.10.0+cu130 with all CUDA 13 dependencies..."
uv pip install --reinstall torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

echo "      -> Reinstalling cuequivariance for CUDA 13..."
uv pip install cuequivariance-ops-cu13 cuequivariance-ops-torch-cu13 2>/dev/null || true

echo "      -> Re-pinning PyTorch Geometric packages for torch 2.10.0+cu130..."
uv pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.10.0+cu130.html --force-reinstall --no-deps

echo "      PyTorch version after re-pin: $(python -c 'import torch; print(torch.__version__)')"

# ------------------------------------------------------------------------------
# Done!
# ------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/.venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo ""
echo "Note: Foldseek and MMseqs2 are not included in the public build."
echo "Install them separately if needed:"
echo "  wget https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz"
echo "  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
