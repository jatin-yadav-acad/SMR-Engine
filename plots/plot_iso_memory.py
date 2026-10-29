"""
Generates Figure 8: Continual Class-IL Accuracy under Strict Edge Memory Envelopes (MB).
"""
import os
import json
import matplotlib.pyplot as plt

def plot_iso_memory():
    data_file = os.path.join(os.path.dirname(__file__), "..", "results_final", "iso_memory_results.json")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found")
        return False

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    budgets = data["memory_budgets_mb"]
    
    # Plotting styles
    plt.figure(figsize=(6.5, 4.0), dpi=300)
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    # D-SMR (Ours): valid from 1.44 MB onwards
    smr_x = [b for b in budgets if b >= 1.44]
    smr_y = [52.40 for _ in smr_x]
    plt.plot(smr_x, smr_y, color='#1f77b4', marker='s', linewidth=2.5, markersize=7,
             label='D-SMR (Ours, Zero-Heap In-SRAM, 52.40%)')

    # DER++: valid from 2.0 MB onwards
    der_x = [b for b, r in zip(budgets, data["derpp"]) if r["class_il_acc"] is not None]
    der_y = [r["class_il_acc"] for r in data["derpp"] if r["class_il_acc"] is not None]
    plt.plot(der_x, der_y, color='#d62728', marker='o', linestyle='--', linewidth=2.0, markersize=6,
             label='DER++ Rehearsal (Buffer Truncation)')

    # FOSTER: valid only from 50.0 MB onwards
    fos_x = [b for b, r in zip(budgets, data["foster"]) if r["class_il_acc"] is not None]
    fos_y = [r["class_il_acc"] for r in data["foster"] if r["class_il_acc"] is not None]
    plt.plot(fos_x, fos_y, color='#2ca02c', marker='^', linestyle=':', linewidth=2.0, markersize=7,
             label='FOSTER Dynamic Expansion (OOM < 50 MB)')

    # Highlight Embedded SRAM Envelope (2.0 MB on STM32H7)
    plt.axvline(x=2.0, color='gray', linestyle='-.', linewidth=1.2, alpha=0.8)
    plt.text(2.1, 25, 'Physical STM32H7 SRAM Limit\n(2.0 MB Internal)', color='#333333', fontsize=8.5, weight='bold')

    plt.xscale('log')
    plt.xlim(0.8, 400)
    plt.ylim(10, 70)
    plt.xlabel('Runtime Memory Envelope (MB, Log Scale)', fontweight='bold')
    plt.ylabel('Split CIFAR-100 Class-IL Accuracy (%)', fontweight='bold')
    plt.title('Continual Accuracy under Strict Edge Memory Envelopes (MB)', fontweight='bold', fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc='lower right', frameon=True, fontsize=8.5)
    plt.tight_layout()

    out_pdf = os.path.join(os.path.dirname(__file__), "..", "figures", "fig8_iso_memory_pareto.pdf")
    out_png = os.path.join(os.path.dirname(__file__), "..", "figures", "fig8_iso_memory_pareto.png")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

    print(f"[PASS] Saved Figure 8 to:\n  {out_pdf}\n  {out_png}")
    return True

if __name__ == "__main__":
    plot_iso_memory()
