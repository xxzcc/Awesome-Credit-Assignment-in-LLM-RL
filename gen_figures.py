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
    fig, ax = plt.subplots(figsize=(24, 11))
    ax.set_xlim(-0.8, 7.6)
    ax.set_ylim(-0.6, 5.2)
    ax.set_aspect('auto')
    ax.axis('off')

    cols = [
        'Monte Carlo', 'TD / Value', 'LLM-as-Critic', 'Game-theoretic',
        'Info-theoretic', 'Uncertainty', 'Verifiable /\nStructured'
    ]
    rows = ['Token', 'Segment', 'Step / Turn', 'Multi-Agent']

    # draw grid
    for i in range(8):  # vertical lines
        ax.plot([i + 0.5 - 0.5, i + 0.5 - 0.5], [0.3, 4.5], color=GRID_COLOR, lw=1)
    for j in range(5):  # horizontal lines
        ax.plot([0, 7.5], [j + 0.5 - 0.2, j + 0.5 - 0.2], color=GRID_COLOR, lw=1)

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
    ax.text(4.0, -0.35, 'Methodology  →', ha='center', va='center', fontsize=12,
            fontweight='bold', color='#374151')

    def draw_cell(x, y, text, bg, edge, tc, fontsize=7.4):
        """Draw one compact taxonomy-cell summary."""
        patch = FancyBboxPatch(
            (x - 0.44, y - 0.35), 0.88, 0.70,
            boxstyle="round,pad=0.05",
            facecolor=bg, edgecolor=edge, linewidth=1.0, zorder=4
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, color=tc, fontweight='bold',
                linespacing=1.18, zorder=5)

    # ---------- Data ----------
    # One compact method group per taxonomy cell keeps the public README figures legible.
    cells = [
        # Token
        (1, 4.0, 'VinePPO', BLUE, BLUE_EDGE, BLUE_TEXT),
        (2, 4.0, 'RED\nT-REG', BLUE, BLUE_EDGE, BLUE_TEXT),
        (4, 4.0, 'From r to Q*', BLUE, BLUE_EDGE, BLUE_TEXT),
        # Segment
        (1, 3.0, 'SPO', BLUE, BLUE_EDGE, BLUE_TEXT),
        (2, 3.0, 'TEMPO', BLUE, BLUE_EDGE, BLUE_TEXT),
        (4, 3.0, 'SCAR', BLUE, BLUE_EDGE, BLUE_TEXT),
        # Step / Turn
        (1, 2.0, 'PURE\nSPRO\nGiGPO\nAT^2PO*', RED, RED_EDGE, RED_TEXT),
        (2, 2.0, 'ArCHer\nAgentPRM\nTurn-PPO\nSORL', RED, RED_EDGE, RED_TEXT),
        (3, 2.0, 'CAPO\nSWEET-RL\nHCAPO\nTARL\nRubric-RL*', RED, RED_EDGE, RED_TEXT, 6.8),
        (4, 2.0, 'C3\nCCPO', RED, RED_EDGE, RED_TEXT),
        (5, 2.0, 'IGPO\nITPO\nA^2TGPO*', RED, RED_EDGE, RED_TEXT),
        (6, 2.0, 'AEM*\nT^2PO*', RED, RED_EDGE, RED_TEXT),
        (7, 2.0, 'GVPO*\nA^3*\nPOAD', RED, RED_EDGE, RED_TEXT),
        # Multi-Agent / related
        (1, 1.0, 'M-GRPO', PURPLE, PURPLE_EDGE, PURPLE_TEXT),
        (3, 1.0, 'LLM-MCA', PURPLE, PURPLE_EDGE, PURPLE_TEXT),
        (4, 1.0, 'QLLM\nCore CA*', PURPLE, PURPLE_EDGE, PURPLE_TEXT),
        (5, 1.0, 'C3†\nSHARP', PURPLE, PURPLE_EDGE, PURPLE_TEXT),
        (6, 1.0, 'Dr. MAS', PURPLE, PURPLE_EDGE, PURPLE_TEXT),
        (7, 1.0, 'MAPPA\nOrch. Traces*', PURPLE, PURPLE_EDGE, PURPLE_TEXT),
    ]

    for cell in cells:
        draw_cell(*cell)

    # Evolution trend arrow
    ax.annotate('', xy=(7.0, 1.25), xytext=(1.2, 3.8),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=2, ls='--',
                                alpha=0.35),
                zorder=1)
    ax.text(4.3, 3.0, 'Evolution trend', ha='center', va='center',
            fontsize=9, fontstyle='italic', color='#9ca3af', rotation=-25,
            alpha=0.65, zorder=1)

    # Legend
    for i, (label, bg, edge, tc) in enumerate([
        ('Reasoning RL', BLUE, BLUE_EDGE, BLUE_TEXT),
        ('Agentic RL', RED, RED_EDGE, RED_TEXT),
        ('Multi-Agent', PURPLE, PURPLE_EDGE, PURPLE_TEXT),
    ]):
        x = 2.5 + i * 1.5
        patch = FancyBboxPatch((x - 0.3, -0.1), 1.2, 0.3,
                               boxstyle="round,pad=0.05",
                               facecolor=bg, edgecolor=edge, linewidth=1)
        ax.add_patch(patch)
        ax.text(x + 0.3, 0.05, label, ha='center', va='center',
                fontsize=9, color=tc, fontweight='bold')

    ax.text(6.7, 0.05, '* recent / adjacent additions', ha='center',
            va='center', fontsize=8, color='#6b7280', fontstyle='italic')

    plt.tight_layout()
    plt.savefig('assets/taxonomy.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved assets/taxonomy.png")


# ============================================================
# Figure 2 — Taxonomy Tree (horizontal)
# ============================================================
def make_taxonomy_tree():
    fig, ax = plt.subplots(figsize=(26, 18))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 32)
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
    root = box(ax, 0.5, 16, 'Credit Assignment\nin LLM RL', '#f3f4f6', '#6b7280', '#111827', fs=11, bold=True)

    # ---- Level 1: Three branches ----
    # Reasoning RL
    r_y_center = 27
    r_box = box(ax, 2.5, r_y_center, 'Reasoning RL', BLUE, BLUE_EDGE, BLUE_TEXT, fs=10, bold=True)
    # Agentic RL
    a_y_center = 15
    a_box = box(ax, 2.5, a_y_center, 'Agentic RL', RED, RED_EDGE, RED_TEXT, fs=10, bold=True)
    # Multi-Agent
    m_y_center = 2.0
    m_box = box(ax, 2.5, m_y_center, 'Multi-Agent', PURPLE, PURPLE_EDGE, PURPLE_TEXT, fs=10, bold=True)

    # Connect root to branches
    for by in [r_y_center, a_y_center, m_y_center]:
        connect_h(ax, root[1], 16, r_box[0], by)

    # ---- Reasoning sub-branches ----
    r_subs = [
        ('Token-Level', 30.0, [
            'VinePPO (MC), RED (Redistrib.)',
            'T-REG (Self-gen.), From r to Q* (Implicit)'
        ]),
        ('Segment-Level', 27.5, [
            'SPO (MC), SCAR (Shapley), TEMPO (Tree-TD)'
        ]),
        ('Step-Level', 25.0, [
            'PURE (PRM), SPRO (Masked Adv.), CAPO (LLM-Critic)',
            'ACPO (Attribution), HICRA (Hierarchy), PRL (Entropy)',
            'InT (Intervention), FinePO (Fine PRM), Rubric-RL*'
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
        ('Turn-Level PRM', 22.0, [
            'AgentPRM (TD+GAE), SWEET-RL (Priv. Critic)',
            'Turn-PPO (Turn MDP), SORL (Bias-corr.)',
            'TARL (LLM-Judge), ITPO (Implicit), Turn-Level (Hybrid)'
        ]),
        ('Hindsight /\nCounterfactual', 19.0, [
            'HCAPO (Hindsight), C3 (LOO)',
            'CCPO (Causal), CriticSearch (Retrospective)'
        ]),
        ('Critic-Free /\nStep-Level', 16.0, [
            'GiGPO (Group-in-Group), POAD (Action Decomp.)',
            'CARL (Entropy), iStar (Implicit DPO)'
        ]),
        ('Uncertainty /\nInformation Gain', 13.0, [
            'IGPO (Info gain), A^2TGPO (Adaptive clipping)',
            'AEM (Entropy modulation), T^2PO (Uncertainty control)'
        ]),
        ('Verifiable /\nStructured Feedback', 10.0, [
            'GVPO (Process-verifiable feedback)',
            'A^3 / CLI agents (Structured action credit)'
        ]),
        ('Hierarchical', 7.0, [
            'ArCHer (TD Hierarchy), PilotRL (Progressive)',
            'StepAgent (IRL), AT^2PO (Turn tree search)'
        ]),
        ('Adjacent\nEnablers', 4.5, [
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
        ('Agent-Level', 2.0, [
            'M-GRPO (Hierarchical), SHARP (Shapley)',
            'MAPPA (Per-action PRM), Dr. MAS (Agent-wise Adv.)',
            'LLM-MCA (LLM-Critic), QLLM (LLM-gen.)',
            'Orchestration Traces* (token-to-team units)',
            'In-Context Core* (coalition/contributor credit)'
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
