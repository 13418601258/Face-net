import subprocess
import os

def run_deep_mvlm(input_mesh_dir, output_dir):
    cmd = [
        "python",
        "Deep-MVLM/inference.py",
        "--input", input_mesh_dir,
        "--output", output_dir
    ]
    subprocess.run(cmd)