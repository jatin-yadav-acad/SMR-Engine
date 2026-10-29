#!/usr/bin/env python
"""
fig1_hero_pipeline.py
=====================
Generates Figure 1: The SMR-Engine Full Systems Architecture Schematic.
Explicitly illustrates the three co-designed planes:
  1. Control Plane: Autonomous Fast-Gate & Top-k Conformal Candidate Router
  2. Memory Plane: Compile-Time Static Execution Arena (Scratchpads, Envelope, Metadata Cache)
  3. Compute Plane: Block-Compacted WMMA Kernel, Static Weight Folding, Forward Recalibration

Outputs:
  - manuscript/figures/fig1_hero_pipeline.pdf
  - manuscript/figures/fig1_hero_pipeline.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

os.makedirs('manuscript/figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8.5,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(10.5, 3.2), dpi=300)
ax.set_xlim(0, 105)
ax.set_ylim(0, 36)
ax.axis('off')

# ==========================================
# 1. CONTROL PLANE (Left Panel: Blue Tint)
# ==========================================
p_ctrl = FancyBboxPatch((1, 1.5), 32.5, 33, boxstyle="round,pad=0.4,rounding_size=1.0",
                        facecolor='#F0F4F8', edgecolor='#3182CE', lw=1.5)
ax.add_patch(p_ctrl)

ax.text(17.25, 32.2, "Control Plane: Autonomous Routing", fontsize=9.2, fontweight='bold',
        color='#1A365D', ha='center', va='center')
ax.text(17.25, 30.2, r"Fast-Gate Centroids & Conformal Candidate Triage", fontsize=6.8,
        color='#4A5568', ha='center', va='center')

# Input x
b_in = FancyBboxPatch((2.2, 18.5), 3.8, 8.5, boxstyle="square,pad=0",
                      facecolor='#EDF2F7', edgecolor='#718096', lw=1.1)
ax.add_patch(b_in)
ax.text(4.1, 22.75, "Input\n$x$", fontsize=7.5, ha='center', va='center', fontweight='bold', color='#2D3748')

ax.annotate('', xy=(7.2, 22.75), xytext=(6.0, 22.75),
            arrowprops=dict(arrowstyle="->", lw=1.1, color='#4A5568'))

# Sensory Backbone
b_sens = FancyBboxPatch((7.2, 15.5), 8.8, 14.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                        facecolor='#2B6CB0', edgecolor='#1A365D', lw=1.1)
ax.add_patch(b_sens)
ax.text(11.6, 26.5, "Sensory", fontsize=7.5, color='white', ha='center', fontweight='bold')
ax.text(11.6, 24.5, "Backbone", fontsize=7.5, color='white', ha='center', fontweight='bold')
ax.text(11.6, 21.0, "conv1..layer3", fontsize=6.2, color='#EBF8FF', ha='center')
ax.text(11.6, 17.5, r"$h_s(x) \in \mathbb{R}^{256}$", fontsize=6.8, color='#BEE3F8', ha='center', fontweight='bold')

ax.annotate('', xy=(17.5, 22.75), xytext=(16.0, 22.75),
            arrowprops=dict(arrowstyle="->", lw=1.1, color='#4A5568'))

# Fast-Gate Box
b_fg = FancyBboxPatch((17.5, 17.5), 14.8, 11.0, boxstyle="round,pad=0.2,rounding_size=0.4",
                      facecolor='#EBF8FF', edgecolor='#2B6CB0', lw=1.1)
ax.add_patch(b_fg)
ax.text(24.9, 26.0, r"$\mathcal{O}(1)$ Fast-Gate ($10.10\,\mathrm{ms}$)", fontsize=7.2, fontweight='bold', color='#1A365D', ha='center')
ax.text(24.9, 23.2, r"Mahalanobis: $d_M^2(h_s, P_t) \leq \hat{q}_\alpha$", fontsize=6.2, color='#2C5282', ha='center')
ax.text(24.9, 20.0, r"Resident Centroids $P_t$ ($204.8\,\mathrm{KB}$)", fontsize=6.0, color='#4A5568', ha='center')

# Conformal Triage Box
b_conf = FancyBboxPatch((17.5, 5.0), 14.8, 10.5, boxstyle="round,pad=0.2,rounding_size=0.4",
                        facecolor='#FFF', edgecolor='#319795', lw=1.1)
ax.add_patch(b_conf)
ax.text(24.9, 13.0, "Conformal Candidate Set", fontsize=7.0, fontweight='bold', color='#234E52', ha='center')
ax.text(24.9, 10.2, r"$\mathcal{C}_\alpha(x) = \{t \mid d_M^2 \leq 8.8039\}$", fontsize=6.2, color='#285E61', ha='center')
ax.text(24.9, 7.5, r"Edge: $|\mathcal{C}| \leq 2$ ($84.7\%$) | Full: $\geq 95\%$", fontsize=5.8, color='#319795', ha='center', fontweight='bold')

ax.annotate('', xy=(24.9, 15.5), xytext=(24.9, 17.5),
            arrowprops=dict(arrowstyle="->", lw=1.1, color='#2B6CB0'))

# Footer badge Control Plane
b_cfoot = FancyBboxPatch((2.2, 2.3), 30.1, 2.2, boxstyle="square,pad=0",
                         facecolor='#BEE3F8', edgecolor='none')
ax.add_patch(b_cfoot)
ax.text(17.25, 3.4, "Autonomous Class-IL Dispatch  |  Zero Replay Exemplars", fontsize=5.8, color='#1A365D', ha='center', va='center', fontweight='bold')

# Arrow Control -> Memory
ax.annotate('', xy=(34.8, 18.0), xytext=(33.5, 18.0),
            arrowprops=dict(arrowstyle="->", lw=1.5, color='#4A5568'))

# ==========================================
# 2. MEMORY PLANE (Middle Panel: Purple Tint)
# ==========================================
p_mem = FancyBboxPatch((35.0, 1.5), 33.5, 33, boxstyle="round,pad=0.4,rounding_size=1.0",
                       facecolor='#FAF5FF', edgecolor='#805AD5', lw=1.5)
ax.add_patch(p_mem)

ax.text(51.75, 32.2, "Memory Plane: Static Execution Arena", fontsize=9.2, fontweight='bold',
        color='#44337A', ha='center', va='center')
ax.text(51.75, 30.2, r"Compile-Time Zero-Heap Guarantee (Min_Heap_Size = 0x0)", fontsize=6.8,
        color='#4A5568', ha='center', va='center')

# Arena Component 1: Ping-Pong Scratchpads
b_sc = FancyBboxPatch((36.2, 20.5), 31.1, 7.8, boxstyle="round,pad=0.2,rounding_size=0.4",
                      facecolor='#FFF', edgecolor='#805AD5', lw=1.1)
ax.add_patch(b_sc)
ax.text(51.75, 26.2, "Dual Ping-Pong Activation Scratchpad", fontsize=7.2, fontweight='bold', color='#553C9A', ha='center')
ax.text(51.75, 23.8, r"GPU: $B=128$, $C=512$  |  MCU: $120\,\mathrm{KB}$ INT8 ($B=1$ in AXI-SRAM)", fontsize=6.0, color='#4A5568', ha='center')
ax.text(51.75, 21.8, "Zero Dynamic Allocation  |  Non-Owning Tensor Views", fontsize=5.8, color='#6B46C1', ha='center', fontweight='bold')

# Arena Component 2: Static Weight Envelope
b_we = FancyBboxPatch((36.2, 11.5), 31.1, 7.8, boxstyle="round,pad=0.2,rounding_size=0.4",
                      facecolor='#FFF', edgecolor='#6B46C1', lw=1.1)
ax.add_patch(b_we)
ax.text(51.75, 17.2, "Static Weight Envelope ($C_{\max}$ Capacity)", fontsize=7.2, fontweight='bold', color='#44337A', ha='center')
ax.text(51.75, 14.8, r"Server VRAM: Dense Base  |  MCU: Flash XIP ($480\,\mathrm{KB}$)", fontsize=6.0, color='#4A5568', ha='center')
ax.text(51.75, 12.8, r"Pre-Allocated Modular Expansion Envelopes (D-SMR $+18.7\%$)", fontsize=5.8, color='#553C9A', ha='center')

# Arena Component 3: Resident Metadata Cache
b_md = FancyBboxPatch((36.2, 5.0), 31.1, 5.6, boxstyle="round,pad=0.2,rounding_size=0.4",
                      facecolor='#FFF', edgecolor='#9F7AEA', lw=1.1)
ax.add_patch(b_md)
ax.text(51.75, 8.8, r"Resident Metadata Cache ($1.44\,\mathrm{MB}$ across 50 Tasks)", fontsize=6.8, fontweight='bold', color='#44337A', ha='center')
ax.text(51.75, 6.4, r"Bitmasks $3.2\,\mathrm{KB}$ | Moments $205\,\mathrm{KB}$ | DMA Paging $102.5\,\mu\mathrm{s}$", fontsize=5.8, color='#6B46C1', ha='center')

# Footer badge Memory Plane
b_mfoot = FancyBboxPatch((36.2, 2.3), 31.1, 2.2, boxstyle="square,pad=0",
                         facecolor='#E9D8FD', edgecolor='none')
ax.add_patch(b_mfoot)
ax.text(51.75, 3.4, "Flat 1.44 MB SRAM Watermark  |  Zero Allocator Contention", fontsize=5.8, color='#44337A', ha='center', va='center', fontweight='bold')

# Arrow Memory -> Compute
ax.annotate('', xy=(69.8, 18.0), xytext=(68.5, 18.0),
            arrowprops=dict(arrowstyle="->", lw=1.5, color='#4A5568'))

# ==========================================
# 3. COMPUTE PLANE (Right Panel: Green Tint)
# ==========================================
p_comp = FancyBboxPatch((70.0, 1.5), 33.5, 33, boxstyle="round,pad=0.4,rounding_size=1.0",
                        facecolor='#F0FFF4', edgecolor='#38A169', lw=1.5)
ax.add_patch(p_comp)

ax.text(86.75, 32.2, "Compute Plane: Zero-DRAM GEMM", fontsize=9.2, fontweight='bold',
        color='#22543D', ha='center', va='center')
ax.text(86.75, 30.2, r"Compacted WMMA Kernel & Static Weight Folding", fontsize=6.8,
        color='#4A5568', ha='center', va='center')

# Compute Component 1: Compacted WMMA Kernel
b_wmma = FancyBboxPatch((71.2, 20.5), 31.1, 7.8, boxstyle="round,pad=0.2,rounding_size=0.4",
                        facecolor='#FFF', edgecolor='#38A169', lw=1.1)
ax.add_patch(b_wmma)
ax.text(86.75, 26.2, "Divergence-Free Compacted WMMA Kernel", fontsize=7.2, fontweight='bold', color='#22543D', ha='center')
ax.text(86.75, 23.8, r"Cooperative 16-channel compaction via __ballot_sync / __popc", fontsize=5.8, color='#4A5568', ha='center')
ax.text(86.75, 21.8, r"$2.48\times\text{--}3.74\times$ speedup over naive branch  |  Line-rate Tensor Cores", fontsize=5.6, color='#276749', ha='center', fontweight='bold')

# Compute Component 2: Static Weight Folding
b_fold = FancyBboxPatch((71.2, 11.5), 31.1, 7.8, boxstyle="round,pad=0.2,rounding_size=0.4",
                        facecolor='#FFF', edgecolor='#2F855A', lw=1.1)
ax.add_patch(b_fold)
ax.text(86.75, 17.2, r"Dual-Regime Weight Folding ($\tilde{W} = \frac{\gamma}{\sigma} W$)", fontsize=7.2, fontweight='bold', color='#1C4532', ha='center')
ax.text(86.75, 14.8, r"Class-IL: Fused register epilogue ($C[c] \cdot s_t + b_t$, 0 ms)", fontsize=5.8, color='#4A5568', ha='center')
ax.text(86.75, 12.8, r"Task-Switch: In-place SRAM ratio rescaling ($s_B/s_A$ in $0.03\,\mathrm{ms}$)", fontsize=5.6, color='#22543D', ha='center', fontweight='bold')

# Compute Component 3: Forward Recalibration
b_recal = FancyBboxPatch((71.2, 5.0), 31.1, 5.6, boxstyle="round,pad=0.2,rounding_size=0.4",
                         facecolor='#FFF', edgecolor='#48BB78', lw=1.1)
ax.add_patch(b_recal)
ax.text(86.75, 8.8, r"Forward-Only Recalibrator ($K=20$, Zero-Backprop)", fontsize=6.8, fontweight='bold', color='#22543D', ha='center')
ax.text(86.75, 6.4, r"ResNet-18: $0.45\,\mathrm{s}$ ($87.72\,\mathrm{J}$)  |  MobileNetV3: $0.68\,\mathrm{s}$", fontsize=5.8, color='#276749', ha='center')

# Footer badge Compute Plane
b_cfoot = FancyBboxPatch((71.2, 2.3), 31.1, 2.2, boxstyle="square,pad=0",
                         facecolor='#C6F6D5', edgecolor='none')
ax.add_patch(b_cfoot)
ax.text(86.75, 3.4, "Dead Channels Revived (88.4%) | 1.61x GPU / 28.54x MCU Energy Cut", fontsize=5.6, color='#22543D', ha='center', va='center', fontweight='bold')

plt.tight_layout()
fig.savefig('manuscript/figures/fig1_hero_pipeline.pdf', bbox_inches='tight', dpi=300)
fig.savefig('manuscript/figures/fig1_hero_pipeline.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print("[SUCCESS] Generated professional Control/Memory/Compute Figure 1!")
