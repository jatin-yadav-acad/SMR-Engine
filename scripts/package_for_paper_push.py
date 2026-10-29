import os
import shutil
import subprocess
import sys

def remove_readonly(func, path, excinfo):
    import stat
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def package_repo():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_code_dir = os.path.join(base_dir, "SMR-Engine")
    target_ms_dir = os.path.join(base_dir, "SMR-Engine-Manuscript")

    print("=" * 80)
    print("  PACKAGING MLSYS 2027 SUBMISSION FOLDERS")
    print(f"  1. Code Repository:       {target_code_dir}")
    print(f"  2. Manuscript Repository: {target_ms_dir}")
    print("=" * 80)

    # =========================================================================
    # PART 1: SMR-Engine (All Codes, Firmware, Benchmarks, Tests, Results)
    # =========================================================================
    print(f"\n[*] Packaging Code Repository into: {target_code_dir}...")
    if os.path.exists(target_code_dir):
        shutil.rmtree(target_code_dir, onerror=remove_readonly)
    os.makedirs(target_code_dir, exist_ok=True)

    # 1. Root code & configuration files
    root_files = [
        "README.md",
        "LICENSE",
        "Dockerfile",
        "docker-compose.yml",
        "setup.py",
        "requirements.txt",
        "environment.yml",
        "pytest.ini",
        "run_all_benchmarks.py",
        "run_all_benchmarks.sh",
        "evaluate.py",
        "kernel.cu",
        "sram_envelope_verification.py",
        "reproduce_normalization_revival.py",
        "reproduce_all.py",
        "reproduce_cifar100.py",
        "reproduce_crossover.py",
        "reproduce_energy.py",
        "reproduce_neural_collapse.py",
        "reproduce_robustness.py",
        "reproduce_spectral_decay.py",
        "reproduce_vit.py",
        "conformal_coverage_eval.py",
        "train_full_continual_suite.py",
        "train_full_continual_suite.sh",
        "setup_datasets.py",
        "setup_imagenet_splits.py",
        "smr_supplementary_code.zip"
    ]

    for rf in root_files:
        src_path = os.path.join(base_dir, rf)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(target_code_dir, rf))
            print(f"  + Copied root file: {rf}")

    # 2. Write clean .gitignore for the target repo
    gitignore_content = """# Python & bytecode
__pycache__/
*.py[cod]
*$py.class
*.so
*.exe
*.o
*.obj
.pytest_cache/
.coverage
htmlcov/

# Environments
venv/
env/
.env/
conda-env/

# System and editor
.DS_Store
Thumbs.db
.vscode/
.idea/
*.swp
*.swo

# LaTeX build intermediates (keep PDFs and figures)
manuscript/*.aux
manuscript/*.log
manuscript/*.out
manuscript/*.synctex.gz
manuscript/*.bbl
manuscript/*.blg
manuscript/*.fls
manuscript/*.fdb_latexmk
manuscript/*.toc

# Raw heavy training checkpoints (weights are deterministic/generated)
checkpoints/
*.pth
*.pt
"""
    with open(os.path.join(target_code_dir, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print("  + Generated clean .gitignore")

    # 3. Synchronize clean directories
    sync_dirs = ["src", "firmware", "tests", "experiments", "plots", "results_final"]
    for d in sync_dirs:
        src_d = os.path.join(base_dir, d)
        if os.path.exists(src_d):
            dst = os.path.join(target_code_dir, d)
            shutil.copytree(src_d, dst, ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", "*.pyo", "*.tmp", ".pytest_cache", "*.exe", "*.o", "*.obj"
            ))
            print(f"  + Synchronized clean directory: {d}")

    # 4. Synchronize scripts (all production scripts)
    os.makedirs(os.path.join(target_code_dir, "scripts"), exist_ok=True)
    prod_scripts = [
        "benchmark_wmma_compaction.py",
        "build_zip.py",
        "check_page_counts.py",
        "package_for_paper_push.py",
        "plot_master_benchmark.py",
        "profile_rtx5070ti.py",
        "profile_ryzen9.py",
        "profile_ryzen9.sh",
        "reproduce_smr_osm.py",
        "run_smr_lora_benchmark.py",
        "run_tinyml_streaming.py",
        "sync_anonymous_repo.py",
        "verify_mcu_renode_model.py",
        "verify_stm32_mcu.py",
        "verify_zero_heap_determinism.py"
    ]
    for ps in prod_scripts:
        src_path = os.path.join(base_dir, "scripts", ps)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(target_code_dir, "scripts", ps))
    print(f"  + Synchronized scripts/ ({len(prod_scripts)} production scripts)")

    # 5. Synchronize manuscript/ inside SMR-Engine (strictly clean sources, figures, compiled PDFs)
    dst_manuscript = os.path.join(target_code_dir, "manuscript")
    os.makedirs(dst_manuscript, exist_ok=True)
    manuscript_clean_files = [
        "manuscript.tex",
        "supplementary.tex",
        "merge.tex",
        "mlsys2027.sty",
        "mlsys2027.bst",
        "mlsys2026.sty",
        "mlsys2026.bst",
        "manuscript.pdf",
        "supplementary.pdf",
        "merge.pdf",
        "full_submission_merged.pdf"
    ]
    for mf in manuscript_clean_files:
        src_path = os.path.join(base_dir, "manuscript", mf)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(dst_manuscript, mf))

    # Copy manuscript/figures/
    src_fig = os.path.join(base_dir, "manuscript", "figures")
    dst_fig = os.path.join(dst_manuscript, "figures")
    if os.path.exists(src_fig):
        shutil.copytree(src_fig, dst_fig)
    print("  + Synchronized SMR-Engine/manuscript/ (clean LaTeX sources, MLSys styles, figures, and compiled PDFs)")

    # Initialize git in SMR-Engine
    try:
        subprocess.run(["git", "init"], cwd=target_code_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "SMR-Engine Author"], cwd=target_code_dir, check=True)
        subprocess.run(["git", "config", "user.email", "author@smr-engine.org"], cwd=target_code_dir, check=True)
        subprocess.run(["git", "add", "."], cwd=target_code_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit: SMR-Engine MLSys 2027 camera-ready release"], cwd=target_code_dir, check=True, capture_output=True)
        print("  [SUCCESS] SMR-Engine git repository initialized with clean commit!")
    except Exception as e:
        print(f"  [WARNING] Git init failed or not available: {e}")

    # =========================================================================
    # PART 2: SMR-Engine-Manuscript (Strictly Manuscript Sources, Figures, PDFs)
    # =========================================================================
    print(f"\n[*] Packaging Manuscript Repository into: {target_ms_dir}...")
    if os.path.exists(target_ms_dir):
        shutil.rmtree(target_ms_dir, onerror=remove_readonly)
    os.makedirs(target_ms_dir, exist_ok=True)

    # Manuscript core files
    ms_files = [
        "manuscript.tex",
        "supplementary.tex",
        "merge.tex",
        "mlsys2027.sty",
        "mlsys2027.bst",
        "mlsys2026.sty",
        "mlsys2026.bst",
        "manuscript.pdf",
        "supplementary.pdf",
        "merge.pdf"
    ]
    for mf in ms_files:
        src_path = os.path.join(base_dir, "manuscript", mf)
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(target_ms_dir, mf))
            print(f"  + Copied manuscript file: {mf}")

    # Also copy merge.pdf as full_submission_merged.pdf in SMR-Engine-Manuscript
    merge_src = os.path.join(base_dir, "manuscript", "merge.pdf")
    if os.path.exists(merge_src):
        shutil.copy2(merge_src, os.path.join(target_ms_dir, "full_submission_merged.pdf"))
        print("  + Copied merge.pdf -> full_submission_merged.pdf")

    # Copy check_page_counts.py if exists
    cpc = os.path.join(base_dir, "scripts", "check_page_counts.py")
    if os.path.exists(cpc):
        shutil.copy2(cpc, os.path.join(target_ms_dir, "check_page_counts.py"))

    # Copy manuscript/figures/
    dst_ms_fig = os.path.join(target_ms_dir, "figures")
    if os.path.exists(src_fig):
        shutil.copytree(src_fig, dst_ms_fig)
    print("  + Synchronized SMR-Engine-Manuscript/figures/")

    # Generate a clean README for SMR-Engine-Manuscript
    ms_readme = """# SMR-Engine Manuscript & Supplementary Material

Camera-ready submission documents and LaTeX sources for:
**SMR-Engine: Deterministic-Memory Continuous Inference and Adaptation via In-SRAM Subspace Compaction** (MLSys 2027).

## Documents Included

| Document | File | Page Budget | Content / Scope |
| :--- | :--- | :---: | :--- |
| **Main Conference Paper** | `manuscript.pdf` | **11 Pages** | Pages 1–10: Main text, system architecture, benchmarks, and artifacts.<br>Page 11: References. |
| **Supplementary Material** | `supplementary.pdf` | **25 Pages** | Complete proofs (Appendices A.1–A.7), ablation studies (B.1–B.7), SMR-LoRA serving case study (C), and extended systems audits (D.1–D.9). |
| **Merged Full Submission** | `full_submission_merged.pdf` | **36 Pages** | Merged camera-ready package (11 + 25 = 36 pages). |

## Compilation Instructions

To compile from LaTeX sources:
```bash
# Main paper (11 pages)
pdflatex manuscript.tex
pdflatex manuscript.tex

# Supplementary material (25 pages)
pdflatex supplementary.tex
pdflatex supplementary.tex

# Merged document (36 pages)
pdflatex merge.tex
```

Verify page counts:
```bash
python check_page_counts.py
```
"""
    with open(os.path.join(target_ms_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(ms_readme)
    print("  + Generated clean SMR-Engine-Manuscript/README.md")

    # Clean any accidental build artifacts
    for root, dirs, files in os.walk(target_ms_dir):
        for f in files:
            if f.endswith((".aux", ".log", ".out", ".synctex.gz", ".bbl", ".blg")):
                os.remove(os.path.join(root, f))

    # Also sync anonymous_github_repo if present
    anon_dir = os.path.join(base_dir, "anonymous_github_repo")
    if os.path.exists(anon_dir):
        anon_ms = os.path.join(anon_dir, "manuscript")
        os.makedirs(anon_ms, exist_ok=True)
        for mf in manuscript_clean_files:
            src_path = os.path.join(base_dir, "manuscript", mf)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(anon_ms, mf))

    print("\n" + "=" * 80)
    print("  [COMPLETE] BOTH FOLDERS ARE PACKAGED AND READY:")
    print("  1. 'SMR-Engine/'           --> Complete Codebase, Firmware, Tests, Scripts, Results")
    print("  2. 'SMR-Engine-Manuscript/'--> Complete Manuscript Sources, Figures, Compiled PDFs")
    print("=" * 80)

if __name__ == "__main__":
    package_repo()
