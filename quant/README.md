# Quantization Pipeline (`quant.sh`)

This script automates the GGUF quantization process using `llama.cpp`. It supports both local staging/packaging and end-to-end execution on an EC2 instance.

## Usage

### 1. Packaging (Local Machine)
Prepare your model files and configuration, then bundle them for the cloud.
```bash
./quant.sh --package
```

### 2. Running (EC2 Instance)
Execute the quantization job on a remote instance.
```bash
./quant.sh --run <job_name>
```

---

## Configuration (`quant.yaml`)

The script expects a `quant.yaml` file in the current directory.

### Schema
| Key | Type | Description |
| :--- | :--- | :--- |
| `name` | string | Unique name for the job (used for folder and tarball naming). |
| `model_dir` | string | Path to the directory containing HF weights (safetensors, config, tokenizer). |
| `quants` | list | List of target quantization types (e.g., `Q6_K`, `Q4_K_M`). |
| `imatrix` | boolean | If `true`, runs `llama-imatrix` using `./imatrix/calibration.txt`. |
| `output_prefix` | string | Filename prefix for the resulting GGUF files. |
| `gpu` | boolean | If `true`, builds `llama.cpp` with CUDA support; otherwise CPU-only. |

---

## How It Works

### Mode: `--package` (Local)
1. **Validation**: Checks that `quant.yaml` exists, `model_dir` contains safetensors, and (if enabled) `imatrix/calibration.txt` is present.
2. **Staging**: Creates a `quant-job-<name>/` directory and copies the script, config, model, and imatrix data into it.
3. **Bundling**: Compresses the staging directory into a `.tar.gz` file.
4. **S3 Upload**: Copies the tarball to the S3 jobs directory (`~/s3/mad-lab-files/quant-jobs/`).

### Mode: `--run` (EC2)
1. **S3 Setup**: Installs `mountpoint-s3` and mounts the `mad-lab-files` bucket.
2. **Unpacking**: Pulls the specified job tarball from S3 and extracts it.
3. **Environment**: Installs system dependencies (Git, CMake, CUDA if requested) and sets up a Python venv.
4. **NVMe Mount**: Detects, formats (if needed), and mounts the first available NVMe disk to `/mnt/nvme` for fast workspace access.
5. **Build**: Clones and builds `llama.cpp` with appropriate optimizations (CUDA or Native CPU).
6. **Pipeline**:
    - Converts HF weights to F16 GGUF.
    - (Optional) Generates Importance Matrix (imatrix).
    - Quantizes into requested formats.
7. **Cleanup & Upload**: Deletes intermediate files and uploads final GGUFs to S3 (`quant-output/`).

## Requirements
- **Local**: Python 3, `pip` (for `pyyaml`).
- **EC2**: Ubuntu/Debian based, S3 IAM permissions for the `mad-lab-files` bucket.
- **S3 Layout**:
    - `quant-jobs/`: Storage for bundled job tarballs.
    - `quant-output/`: Destination for completed GGUF files.
