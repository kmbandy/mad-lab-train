#!/bin/bash
# EC2 setup for Qwen3-1.7B fine-tune
# AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
# Instance: g6.xlarge spot (L4 24GB, sm_89, BF16 native)
# SSH user: ubuntu (NOT ec2-user)
#
# Usage:
#   scp -i ~/.ssh/mad-lab-key.pem -r ~/mad-lab-scripts/quant-finetune/ ubuntu@<IP>:
#   ssh -i ~/.ssh/mad-lab-key.pem ubuntu@<IP>
#   bash quant-finetune/ec2_setup.sh

set -e

sudo apt update -q
sudo apt install -y python3.12-venv

python3 -m venv ~/venv
source ~/venv/bin/activate

pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
pip install --quiet \
    "transformers>=4.51.0" \
    "trl>=0.12.0" \
    "peft>=0.13.0" \
    "bitsandbytes>=0.46.1" \
    "liger-kernel==0.4.2" \
    accelerate \
    datasets

echo ""
echo "Setup complete. Run:"
echo "  source ~/venv/bin/activate"
echo "  cd ~/quant-finetune"
echo "  python3 qwen3_finetune.py --domain technical 2>&1 | tee technical.log"
echo "  python3 qwen3_finetune.py --domain sentiment 2>&1 | tee sentiment.log"
