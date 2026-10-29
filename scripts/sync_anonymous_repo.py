import os
import shutil

def sync_repo():
    src_dir = "."
    dst_dir = "anonymous_github_repo"

    root_files = [
        "evaluate.py",
        "README.md",
        "LICENSE",
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "environment.yml",
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
        "conformal_coverage_eval.py",
        "train_full_continual_suite.py",
        "train_full_continual_suite.sh",
        "kernel.cu",
        "manuscript.pdf",
        "supplementary.pdf",
        "full_submission_merged.pdf",
        "pytest.ini",
    ]

    for rf in root_files:
        if os.path.exists(rf):
            shutil.copy2(rf, os.path.join(dst_dir, rf))
            print(f"Copied root file: {rf}")

    sync_dirs = ["src", "firmware", "tests", "plots", "results_final", "scripts"]
    for d in sync_dirs:
        dst = os.path.join(dst_dir, d)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(d, dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.tmp", ".pytest_cache"
        ))
        print(f"Synchronized directory: {d}")

    # Synchronize manuscript
    dst_manuscript = os.path.join(dst_dir, "manuscript")
    os.makedirs(dst_manuscript, exist_ok=True)
    manuscript_files = [
        "manuscript.pdf",
        "supplementary.pdf",
        "full_submission_merged.pdf",
        "manuscript.tex",
        "supplementary.tex",
        "mlsys2027.sty",
        "mlsys2027.bst",
        "mlsys2026.sty",
        "mlsys2026.bst",
        "merge.tex"
    ]
    for mf in manuscript_files:
        src_path = os.path.join("manuscript", mf)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(dst_manuscript, mf))
            print(f"Copied manuscript file: {mf}")

    # Also copy figures directory in manuscript
    src_fig = os.path.join("manuscript", "figures")
    dst_fig = os.path.join(dst_manuscript, "figures")
    if os.path.exists(src_fig):
        if os.path.exists(dst_fig):
            shutil.rmtree(dst_fig)
        shutil.copytree(src_fig, dst_fig)
        print("Synchronized manuscript/figures")

    print("\n[SUCCESS] Anonymous repository fully synchronized!")

if __name__ == "__main__":
    sync_repo()
