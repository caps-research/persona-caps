"""
Figure 1: Persona-CAPS framework diagram (clean rewrite, no overlapping).
Vertical pipeline with side branches; outputs PDF + PNG preview.

Sizing note (why the canvas is narrow and tall): box geometry is in data units, so
a box's final width is (data fraction x \\columnwidth) regardless of figsize -- but
font sizes are absolute points and shrink by \\columnwidth/figsize_width on the way
in. R2 flagged the text as unreadable, and the fix is a narrower canvas: the same
labels grow by exactly the ratio of the widths.

What limits the narrowing is not the long sentences, as one would guess, but the
*small* boxes. A 7.5 pt "CN-high" label is ~30 bp wide and its corpus cell was
only 9.5 data units, which pinned the canvas at ~328 bp and capped the gain at 5%.
The cells are therefore wider here, the two-word boxes carry their text on two
lines, and the retriever/LLM boxes were stretched -- which moves the binding
constraint down to ~260 bp and buys a third more font.

Height is deliberately generous: gaps are in data units and grow with the canvas
while text height does not, so a taller canvas both lengthens the arrows (R2's
other complaint) and adds clearance between stacked lines rather than removing it.

Narrowing bottoms out where a label starts touching its own box, so that is the
quantity to watch rather than the width: tools/check_figure_clearance.py measures
it and compares against the committed figure. Two changes here came out of that
measurement -- the cells went to 12.8 units with the grid shifted left to keep the
dashed arrow its length, and the pi-box labels were centred on the box (they sat
1.05 units left of centre, which was spending the left margin for nothing).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# matplotlib defaults to Type 3, which carries no Unicode map: the figure text
# then cannot be searched, copied, or read by a screen reader in the published
# PDF. Type 42 embeds the same glyphs as TrueType with a ToUnicode map.
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Chosen so the trimmed PDF lands near 260 x 350 bp: narrow enough that the
# binding text constraint above is just satisfied, tall enough for long arrows.
FIG_W_IN, FIG_H_IN = 4.02, 5.06
Y_LO, Y_HI = -8, 100

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(Y_LO, Y_HI)
ax.axis('off')

# Color palette: B&W with one accent (muted blue) for π/control surface
BOX_STROKE = '#333333'
BOX_FILL = '#ffffff'
ACCENT = '#2c5f8d'      # muted blue (control surface)
ACCENT_LIGHT = '#e8eff5'
AUDIT = '#a0522d'        # muted sienna (audit trail)
AUDIT_LIGHT = '#f5ebe1'
TEXT = '#1a1a1a'


# ---------------------------------------------------------------------------
# Helper: draw a labeled box
# ---------------------------------------------------------------------------
def box(x, y, w, h, text, fontsize=8.5, stroke=BOX_STROKE, fill=BOX_FILL,
        weight='normal', radius=0.012):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius*100}",
                           linewidth=1.0, edgecolor=stroke, facecolor=fill)
    ax.add_patch(patch)
    if text:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=TEXT, weight=weight)


def arrow(x1, y1, x2, y2, label=None, label_offset=(1, 0), color=BOX_STROKE,
          style='-|>', mut=12, lw=1.2, label_fs=7.5):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=mut,
                        color=color, linewidth=lw)
    ax.add_patch(a)
    if label:
        mx, my = (x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1]
        ax.text(mx, my, label, fontsize=label_fs, color=TEXT, ha='left',
                va='center', style='italic')


# ---------------------------------------------------------------------------
# 1. Persona corpus C (top)
# ---------------------------------------------------------------------------
ax.text(50, 96.5, 'Persona Corpus C', ha='center', va='center',
        fontsize=10.5, weight='bold', color=TEXT)
ax.text(50, 92, '166 if-then atoms · ESM-mined · Beck & Jackson (2022)',
        ha='center', va='center', fontsize=7.5, color='#555555', style='italic')

# 10 cells in a 2-row grid
cell_y_high = 82.5
cell_y_low = 75
cell_w, cell_h = 12.8, 5.5
cell_x_start = 11.0
dims = ['N', 'E', 'C', 'A', 'O']
for i, d in enumerate(dims):
    cx = cell_x_start + i * (cell_w + 1.0)
    box(cx, cell_y_high, cell_w, cell_h, f'C{d}-high', fontsize=7.5,
        fill=ACCENT_LIGHT, stroke=ACCENT)
    box(cx, cell_y_low, cell_w, cell_h, f'C{d}-low', fontsize=7.5,
        fill=ACCENT_LIGHT, stroke=ACCENT)

# π control label: a small box on the right linking to the corpus
box(85.0, 74, 14.5, 17, '', fill=ACCENT_LIGHT, stroke=ACCENT, radius=0.012)
ax.text(92.25, 87.3, r'$\pi_{d,p}$', ha='center', va='center', fontsize=11,
        color=ACCENT, weight='bold')
ax.text(92.25, 83.2, 'mixture', ha='center', va='center', fontsize=7,
        color=ACCENT)
ax.text(92.25, 80.0, 'weights', ha='center', va='center', fontsize=7,
        color=ACCENT)
ax.text(92.25, 76.6, '(deployer)', ha='center', va='center', fontsize=6.2,
        color=ACCENT, style='italic')
# connecting arrow from π box to the corpus cell grid
a_pi = FancyArrowPatch((84.8, 81), (79.4, 81),
                       arrowstyle='-|>', mutation_scale=10,
                       color=ACCENT, linewidth=1.0, linestyle='--')
ax.add_patch(a_pi)

# ---------------------------------------------------------------------------
# 2. Sampling arrow → cell z
# ---------------------------------------------------------------------------
arrow(50, 74.5, 50, 70.5, label=r' $z_d \sim \mathrm{Cat}(\pi_d)$',
      label_offset=(1.2, 0.6), color=ACCENT, lw=1.3)

# ---------------------------------------------------------------------------
# 3. Two parallel inputs: situation s | selected cell z (briefer)
# ---------------------------------------------------------------------------
ax.text(22, 69.4, '(query at inference time)', fontsize=6.5,
        color='#666', ha='center', style='italic')
ax.text(73, 69.4, '(from corpus)', fontsize=6.5,
        color='#666', ha='center', style='italic')

# Situation s (left) -- two lines, so the box no longer sets the canvas width
box(3, 58.5, 38, 8.5, 'Situation s\n"a friend cancels"', fontsize=8)
# Selected cell (right)
box(54, 58.5, 38, 8.5, r'Selected cells $\{C_z\}$', fontsize=8)

# Both flow into retriever
arrow(22, 58.3, 38, 52.5, lw=1.1)
arrow(73, 58.3, 62, 52.5, lw=1.1)

# ---------------------------------------------------------------------------
# 4. Retriever
# ---------------------------------------------------------------------------
box(13, 41.5, 74, 10.5, '')
ax.text(50, 49.3, 'Retriever', ha='center', va='center',
        fontsize=9, weight='bold', color=TEXT)
ax.text(50, 45.8, r'all-MiniLM-L6-v2 · cosine · TopK ($k{=}4$)',
        ha='center', va='center', fontsize=7.5, color=TEXT)
ax.text(50, 42.8, 'situation-only encoding (§3)', ha='center', va='center',
        fontsize=6.5, color='#666', style='italic')

arrow(50, 41.3, 50, 37.5, lw=1.2)

# ---------------------------------------------------------------------------
# 5. Retrieved atoms σ(R)
# ---------------------------------------------------------------------------
box(2, 17.5, 96, 19.5, '', radius=0.008)
ax.text(50, 34.6, r'Retrieved atoms $\sigma(R)$ $\to$ system prompt',
        ha='center', va='center', fontsize=8, weight='bold', color=TEXT)
atom_lines = [
    '• WHEN a small inconvenience hits → I dwell, ruminate',
    '• WHEN plans change unexpectedly → my mood drops',
    '• WHEN alone in the evening → I replay it in my head',
    '• WHEN unsure → I assume the worst',
]
for i, line in enumerate(atom_lines):
    ax.text(5.0, 31.0 - i*3.1, line, fontsize=7, color=TEXT, ha='left',
            family='monospace')

ax.text(50, 18.9, '↳ no trait labels in the prompt — personality emerges from the atoms',
        ha='center', va='center', fontsize=6.5, color=ACCENT, style='italic')

# arrow with σ(·) label
arrow(50, 17.3, 50, 13.5, label=r' $\sigma(\cdot)$ render',
      label_offset=(1.2, 0.4), lw=1.2, label_fs=7)

# ---------------------------------------------------------------------------
# 6. LLM
# ---------------------------------------------------------------------------
box(13, 5.0, 74, 8.0, '')
ax.text(50, 10.9, 'LLM (T = 0.7)', ha='center', va='center',
        fontsize=9, weight='bold', color=TEXT)
ax.text(50, 7.3, 'MiniMax · GPT-4o · Gemini · Qwen · L8B',
        ha='center', va='center', fontsize=7.5, color=TEXT)

arrow(50, 4.8, 50, 1.5, lw=1.2)

# ---------------------------------------------------------------------------
# 7. Response y  ↔  Audit trail (bi-directional)
# ---------------------------------------------------------------------------
box(1, -9.5, 45, 10.5, 'Response y\n(scored on Mini-IPIP / PB)', fontsize=7.5)

box(54, -9.5, 45, 10.5, '', fill=AUDIT_LIGHT, stroke=AUDIT)
ax.text(76.5, -1.6, 'Audit trail', ha='center', va='center',
        fontsize=8, weight='bold', color=AUDIT)
ax.text(76.5, -6.2, 'fired atoms $\\to$\nESM provenance row',
        ha='center', va='center', fontsize=6.8, color=AUDIT,
        linespacing=1.35)

# bidirectional arrow between response and audit
a_bi = FancyArrowPatch((46.5, -4.2), (53.5, -4.2),
                       arrowstyle='<->', mutation_scale=7,
                       color=AUDIT, linewidth=1.1)
ax.add_patch(a_bi)

# ---------------------------------------------------------------------------
# Bottom equation strip
# ---------------------------------------------------------------------------
ax.text(50, -13.2,
        r'$z_d \sim \mathrm{Cat}(\pi_d)$,    '
        r'$R \sim P_\tau(R\,|\,q, C_{z_d}) \propto e^{\,\mathrm{sim}(q,R)/\tau}$,    '
        r'$y \sim P_\theta(y\,|\,q, \sigma(R))$',
        ha='center', va='center', fontsize=7.5, color=TEXT)

# Extend ylim down to accommodate the response row and the equation
ax.set_ylim(-16, 100)

plt.tight_layout()
plt.savefig('figure1_framework.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('figure1_framework.png', bbox_inches='tight', pad_inches=0.05,
            dpi=180)
print("Figure 1 saved.")
