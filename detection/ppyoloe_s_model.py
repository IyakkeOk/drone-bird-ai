import os
import subprocess


PADDLE_DET_DIR = "PaddleDetection"
CONFIG_FILE = "configs/fbd_sv_2024_ppyoloe.yaml"
OUTPUT_DIR = "outputs/ppyoloe_s"


def run(cmd):
    print(f"[RUNNING]: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def setup_paddledetection():
    if not os.path.exists(PADDLE_DET_DIR):
        run("git clone https://github.com/PaddlePaddle/PaddleDetection.git")

    os.chdir(PADDLE_DET_DIR)
    run("pip install -r requirements.txt")
    run("python setup.py install")
    os.chdir("..")


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run(
        f"cd PaddleDetection && "
        f"python tools/train.py "
        f"-c ../{CONFIG_FILE} "
        f"--eval "
        f"-o output_dir=../{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    setup_paddledetection()
    train()
