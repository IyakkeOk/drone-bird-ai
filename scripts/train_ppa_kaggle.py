import subprocess

# Config file path
config_file = 'configs/fbd_sv_2024_ppyoloe_kaggle.yaml'

# Output directory
output_dir = '/kaggle/working/outputs/ppyoloe_s'

# Run PaddleDetection training
cmd = f"""
cd PaddleDetection && \
python tools/train.py -c ../{config_file} --eval -o output_dir={output_dir}
"""

print("Launching training...")
subprocess.run(cmd, shell=True, check=True)
print("Training finished!")
