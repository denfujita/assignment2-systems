import modal 
import subprocess

app = modal.App("triton-testing")

def build_image(*, include_tests: bool = False) -> modal.Image:
    image = (
            modal.Image.debian_slim(python_version="3.12")
            .run_commands(
                "apt-get update && apt-get install -y wget",
                "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
                "dpkg -i cuda-keyring_1.1-1_all.deb",
                "apt-get update",
            )
            .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
            .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
            .uv_sync()
            .add_local_python_source("cs336_systems")
        )
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image

image = build_image(include_tests=True)

@app.function(image=image, gpu="B200")
def modal_main():
    cmd = ["python", "-m", "pytest", "tests/test_attention.py", "-k", "test_flash_forward_pass_triton"]
    subprocess.run(cmd, cwd="/root", check=True)

@app.local_entrypoint()
def main(test_path="tests/test_attention.py", keyword="test_flash_forward_pass_triton", verbose=True):
    modal_main.remote()
    

