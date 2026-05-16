#!/usr/bin/env python3
"""Generate taxonomy figures for the Awesome-Credit-Assignment-in-LLM-RL repository."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ============================================================
# Color scheme
# ============================================================
BLUE = '#dbeafe'       # reasoning RL bg
BLUE_EDGE = '#3b82f6'  # reasoning RL border
BLUE_TEXT = '#1e40af'   # reasoning RL text

RED = '#fee2e2'        # agentic RL bg
RED_EDGE = '#ef4444'
RED_TEXT = '#991b1b'

PURPLE = '#ede9fe'     # multi-agent bg
PURPLE_EDGE = '#8b5cf6'
PURPLE_TEXT = '#5b21b6'

GRID_COLOR = '#e5e7eb'


def draw_label(ax, x, y, text, color_bg, color_edge, color_text, fontsize=8):
    """Draw a rounded-rect label centered at (x, y)."""
    txt = ax.text(x, y, text, ha='center', va='center',
                  fontsize=fontsize, color=color_text, fontweight='bold',
                  zorder=5)
    # get bbox in display coords, convert to data coords for patch
    ax.figure.canvas.draw()
    bb = txt.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    bb_data = bb.transformed(ax.transData.inverted())
    pad_x, pad_y = 0.12, 0.06
    patch = FancyBboxPatch(
        (bb_data.x0 - pad_x, bb_data.y0 - pad_y),
        bb_data.width + 2*pad_x, bb_data.height + 2*pad_y,
        boxstyle="round,pad=0.05",
        facecolor=color_bg, edgecolor=color_edge, linewidth=0.8,
        zorder=4
    )
    ax.add_patch(patch)


# ============================================================
# Figure 1 — Taxonomy Grid
# ============================================================
def make_taxonomy_grid():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(-0.8, 5.5)
    ax.set_ylim(-0.6, 5.2)
    ax.set_aspect('auto')
    ax.axis('off')

    cols = ['Monte Carlo', 'TD / Value', 'LLM-as-Critic', 'Game-theoretic', 'Info-theoretic']
    rows = ['Token', 'Segment', 'Step / Turn', 'Multi-Agent']

    # draw grid
    for i in range(6):  # vertical lines
        ax.plot([i + 0.5 - 0.5, i + 0.5 - 0.5], [0.3, 4.5], color=GRID_COLOR, lw=1)
    for j in range(5):  # horizontal lines
        ax.plot([0, 5.5], [j + 0.5 - 0.2, j + 0.5 - 0.2], color=GRID_COLOR, lw=1)

    # column headers
    for i, col in enumerate(cols):
        ax.text(i + 1, 4.75, col, ha='center', va='center', fontsize=11, fontweight='bold')

    # row headers
    for j, row in enumerate(rows):
        y = 4 - j
        ax.text(-0.3, y, row, ha='right', va='center', fontsize=11, fontweight='bold')

    # Axis labels
    ax.text(-0.7, 2.5, 'Granularity  ↑', ha='center', va='center', fontsize=12,
            fontweight='bold', rotation=90, color='#374151')
    ax.text(3.0, -0.35, 'Methodology  →', ha='center', va='center', fontsize=12,
            fontweight='bold', color='#374151')

    # ---------- Data ----------
    # (col_idx 1-5, row_y 4=Token, 3=Segment, 2=Step/Turn, 1=Multi-Agent)
    # Token row
    entries = [
        # Token
        (1, 4.0, 'VinePPO', 'r'),
        (2, 4.1, 'RED', 'r'),
        (2, 3.9, 'T-REG', 'r'),
        (4, 4.0, 'From r to Q*', 'r'),
        # Segment
        (1, 3.0, 'SPO', 'r'),
        (2, 3.0, 'TEMPO', 'r'),
        (4, 3.0, 'SCAR', 'r'),
        # Step/Turn — MC
        (1, 2.25, 'PURE', 'r'),
        (1, 2.0, 'GiGPO', 'a'),
        (1, 1.75, 'SPRO', 'r'),
        # Step/Turn — TD
        (2, 2.4, 'ArCHer', 'a'),
        (2, 2.2, 'AgentPRM', 'a'),
        (2, 2.0, 'SPA-RL', 'a'),
        (2, 1.8, 'iStar', 'a'),
        (2, 1.6, 'Turn-PPO', 'a'),
        (1.55, 1.55, 'SORL', 'a'),
        # Step/Turn — LLM-as-Critic
        (3, 2.35, 'CAPO', 'r'),
        (3, 2.1, 'SWEET-RL', 'a'),
        (3, 1.85, 'HCAPO', 'a'),
        (3, 1.6, 'LaRe', 'a'),
        (3.4, 1.55, 'TARL', 'a'),
        # Step/Turn — Game-theoretic
        (4, 2.15, 'C3', 'a'),
        (4, 1.9, 'CCPO', 'a'),
        # Step/Turn — Info-theoretic
        (5, 2.25, 'IGPO', 'a'),
        (5, 2.0, 'CARL', 'a'),
        (5, 1.75, 'HICRA', 'r'),
        (4.5, 1.55, 'ITPO', 'a'),
        # Multi-Agent
        (1, 1.0, 'M-GRPO', 'm'),
        (3, 1.0, 'LLM-MCA', 'm'),
        (4, 1.0, 'QLLM', 'm'),
        (5, 1.0, 'C3†', 'm'),
    ]

    # first pass to render so get_window_extent works
    fig.canvas.draw()

    for (x, y, name, cat) in entries:
        if cat == 'r':
            draw_label(ax, x, y, name, BLUE, BLUE_EDGE, BLUE_TEXT, fontsize=8.5)
        elif cat == 'a':
            draw_label(ax, x, y, name, RED, RED_EDGE, RED_TEXT, fontsize=8.5)
        else:
            draw_label(ax, x, y, name, PURPLE, PURPLE_EDGE, PURPLE_TEXT, fontsize=8.5)

    # Evolution trend arrow
    ax.annotate('', xy=(4.8, 1.2), xytext=(1.2, 3.8),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=2, ls='--'))
    ax.text(3.2, 2.9, 'Evolution trend', ha='center', va='center',
            fontsize=9, fontstyle='italic', color='#9ca3af', rotation=-35)

    # Legend
    for i, (label, bg, edge, tc) in enumerate([
        ('Reasoning RL', BLUE, BLUE_EDGE, BLUE_TEXT),
        ('Agentic RL', RED, RED_EDGE, RED_TEXT),
        ('Multi-Agent', PURPLE, PURPLE_EDGE, PURPLE_TEXT),
    ]):
        x = 1.5 + i * 1.5
        patch = FancyBboxPatch((x - 0.3, -0.1), 1.2, 0.3,
                               boxstyle="round,pad=0.05",
                               facecolor=bg, edgecolor=edge, linewidth=1)
        ax.add_patch(patch)
        ax.text(x + 0.3, 0.05, label, ha='center', va='center',
                fontsize=9, color=tc, fontweight='bold')

    plt.tight_layout()
    plt.savefig('assets/taxonomy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved assets/taxonomy.png")


# ============================================================
# Figure 2 — Taxonomy Tree (horizontal)
# ============================================================
def make_taxonomy_tree():
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 28)
    ax.axis('off')

    # Helper
    def box(ax, x, y, text, bg, edge, tc, fs=8, bold=False):
        fw = 'bold' if bold else 'normal'
        txt = ax.text(x, y, text, ha='center', va='center',
                      fontsize=fs, color=tc, fontweight=fw, zorder=5)
        fig.canvas.draw()
        bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
        bb_d = bb.transformed(ax.transData.inverted())
        px, py = 0.15, 0.08
        p = FancyBboxPatch((bb_d.x0 - px, bb_d.y0 - py),
                           bb_d.width + 2*px, bb_d.height + 2*py,
                           boxstyle="round,pad=0.06",
                           facecolor=bg, edgecolor=edge, linewidth=1, zorder=4)
        ax.add_patch(p)
        return (bb_d.x0 - px, bb_d.x0 + bb_d.width + px,
                bb_d.y0 - py, bb_d.y0 + bb_d.height + py)

    def hline(ax, x1, x2, y, color='#9ca3af'):
        ax.plot([x1, x2], [y, y], color=color, lw=1.2, zorder=2)

    def vline(ax, x, y1, y2, color='#9ca3af'):
        ax.plot([x, x], [y1, y2], color=color, lw=1.2, zorder=2)

    def connect_h(ax, x_from, y_from, x_to, y_to, color='#9ca3af'):
        mid = (x_from + x_to) / 2
        ax.plot([x_from, mid], [y_from, y_from], color=color, lw=1.2, zorder=2)
        ax.plot([mid, mid], [y_from, y_to], color=color, lw=1.2, zorder=2)
        ax.plot([mid, x_to], [y_to, y_to], color=color, lw=1.2, zorder=2)

    fig.canvas.draw()

    # ---- Root ----
    root = box(ax, 0.5, 14, 'Credit Assignment\nin LLM RL', '#f3f4f6', '#6b7280', '#111827', fs=11, bold=True)

    # ---- Level 1: Three branches ----
    # Reasoning RL
    r_y_center = 24
    r_box = box(ax, 2.5, r_y_center, 'Reasoning RL', BLUE, BLUE_EDGE, BLUE_TEXT, fs=10, bold=True)
    # Agentic RL
    a_y_center = 13
    a_box = box(ax, 2.5, a_y_center, 'Agentic RL', RED, RED_EDGE, RED_TEXT, fs=10, bold=True)
    # Multi-Agent
    m_y_center = 1.5
    m_box = box(ax, 2.5, m_y_center, 'Multi-Agent', PURPLE, PURPLE_EDGE, PURPLE_TEXT, fs=10, bold=True)

    # Connect root to branches
    for by in [r_y_center, a_y_center, m_y_center]:
        connect_h(ax, root[1], 14, r_box[0], by)

    # ---- Reasoning sub-branches ----
    r_subs = [
        ('Token-Level', 26.5, [
            'VinePPO (MC), RED (Redistrib.)',
            'T-REG (Self-gen.), From r to Q* (Implicit)'
        ]),
        ('Segment-Level', 24.0, [
            'SPO (MC), SCAR (Shapley), TEMPO (Tree-TD)'
        ]),
        ('Step-Level', 21.5, [
            'PURE (PRM), SPRO (Masked Adv.), CAPO (LLM-Critic)',
            'ACPO (Attribution), HICRA (Hierarchy), PRL (Entropy)',
            'InT (Intervention), FinePO (Fine PRM)'
        ]),
    ]
    for name, y, leaves in r_subs:
        sb = box(ax, 4.5, y, name, '#eff6ff', BLUE_EDGE, BLUE_TEXT, fs=9, bold=True)
        connect_h(ax, r_box[1], r_y_center, sb[0], y)
        leaf_text = '\n'.join(leaves)
        lb = box(ax, 7.8, y, leaf_text, BLUE, BLUE_EDGE, BLUE_TEXT, fs=7.5)
        hline(ax, sb[1], lb[0], y, BLUE_EDGE)

    # ---- Agentic sub-branches ----
    a_subs = [
        ('Turn-Level PRM', 18.5, [
            'AgentPRM (TD+GAE), SWEET-RL (Priv. Critic)',
            'Turn-PPO (Turn MDP), SORL (Bias-corr.)',
            'TARL (LLM-Judge), ITPO (Implicit), Turn-Level (Hybrid)'
        ]),
        ('Hindsight /\nCounterfactual', 15.5, [
            'HCAPO (Hindsight), C3 (LOO)',
            'CCPO (Causal), CriticSearch (Retrospective)'
        ]),
        ('Critic-Free /\nStep-Level', 12.5, [
            'GiGPO (Group-in-Group), POAD (Action Decomp.)',
            'CARL (Entropy), iStar (Implicit DPO), IGPO (Info-theoretic)'
        ]),
        ('Hierarchical', 10.0, [
            'ArCHer (TD Hierarchy), PilotRL (Progressive)',
            'StepAgent (IRL)'
        ]),
        ('Adjacent\nEnablers', 7.5, [
            'SPA-RL, Lightning, RAGEN',
            'SCRIBE, PRS, AdaptSeg'
        ]),
    ]
    for name, y, leaves in a_subs:
        sb = box(ax, 4.5, y, name, '#fef2f2', RED_EDGE, RED_TEXT, fs=9, bold=True)
        connect_h(ax, a_box[1], a_y_center, sb[0], y)
        leaf_text = '\n'.join(leaves)
        lb = box(ax, 7.8, y, leaf_text, RED, RED_EDGE, RED_TEXT, fs=7.5)
        hline(ax, sb[1], lb[0], y, RED_EDGE)

    # ---- Multi-Agent sub-branches ----
    m_subs = [
        ('Agent-Level', 1.5, [
            'M-GRPO (Hierarchical), SHARP (Shapley)',
            'MAPPA (Per-action PRM), Dr. MAS (Agent-wise Adv.)',
            'LLM-MCA (LLM-Critic), QLLM (LLM-gen.)'
        ]),
    ]
    for name, y, leaves in m_subs:
        sb = box(ax, 4.5, y, name, '#f5f3ff', PURPLE_EDGE, PURPLE_TEXT, fs=9, bold=True)
        connect_h(ax, m_box[1], m_y_center, sb[0], y)
        leaf_text = '\n'.join(leaves)
        lb = box(ax, 7.8, y, leaf_text, PURPLE, PURPLE_EDGE, PURPLE_TEXT, fs=7.5)
        hline(ax, sb[1], lb[0], y, PURPLE_EDGE)

    plt.tight_layout()
    plt.savefig('assets/taxonomy_tree.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved assets/taxonomy_tree.png")


if __name__ == '__main__':
    make_taxonomy_grid()
    make_taxonomy_tree()
    print("Done!")
