import os
import zipfile

def build_zip():
    zip_name = "smr_supplementary_code.zip"
    include_dirs = ["src", "experiments", "tests", "plots", "results_final", "firmware", "scripts", "manuscript"]
    include_root_files = [
        "conformal_coverage_eval.py",
        "evaluate.py",
        "run_all_benchmarks.py",
        "run_all_benchmarks.sh",
        "reproduce_all.py",
        "reproduce_cifar100.py",
        "reproduce_crossover.py",
        "reproduce_energy.py",
        "reproduce_neural_collapse.py",
        "reproduce_normalization_revival.py",
        "reproduce_robustness.py",
        "reproduce_spectral_decay.py",
        "reproduce_vit.py",
        "sram_envelope_verification.py",
        "kernel.cu",
        "train_full_continual_suite.py",
        "train_full_continual_suite.sh",
        "requirements.txt",
        "environment.yml",
        "Dockerfile",
        "docker-compose.yml",
        "LICENSE",
        "README.md",
        "manuscript.pdf",
        "supplementary.pdf",
        "full_submission_merged.pdf",
        "pytest.ini"
    ]
    exclude_ext = {".pyc", ".pyo", ".pt", ".pth", ".bin", ".safetensors", ".aux", ".log", ".out", ".synctex.gz", ".fls", ".fdb_latexmk"}
    exclude_dir = {"__pycache__", ".pytest_cache", ".git", "checkpoints", "scratch", "anonymous_github_repo"}

    print(f"[*] Building {zip_name}...")
    count = 0
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in include_root_files:
            if os.path.isfile(f):
                zf.write(f, arcname=f)
                count += 1
        for d in include_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, files in os.walk(d):
                dirs[:] = [s for s in dirs if s not in exclude_dir]
                for f in files:
                    if os.path.splitext(f)[1].lower() in exclude_ext:
                        continue
                    full = os.path.join(root, f)
                    zf.write(full, arcname=os.path.relpath(full, "."))
                    count += 1

    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"[*] Successfully built {zip_name}: {count} files, {size_mb:.2f} MB")

if __name__ == "__main__":
    build_zip()
