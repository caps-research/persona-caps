"""
Figure 2: §4.3 debug-and-fix case study. Shows CAPS's unique interface property:
fired-atom log → corpus row localization → 2-row edit → direction flip.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# See make_figure1.py: Type 3 (the matplotlib default) has no Unicode map, so the
# figure text is unsearchable and inaccessible in the published PDF.
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Inserted at width=\columnwidth, so the render scale is 219 bp / canvas width and
# the height only decides the page cost. The canvas is therefore as short as the
# text allows: gaps are in data units and shrink with it while text height does
# not, so shortening trades box padding and arrow length for a taller page budget
# at no cost in font size. 5.3 in is where the tightest text-to-border clearance
# lands at 2.5 bp; below that the descenders start touching the box strokes.
fig, ax = plt.subplots(figsize=(4.5, 5.3), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Palette
BOX_STROKE = '#333333'
BOX_FILL = '#ffffff'
WRONG = '#b03030'
WRONG_LIGHT = '#fbebeb'
RIGHT_C = '#2d7a3e'
RIGHT_LIGHT = '#e8f3eb'
EDIT = '#7a5d2d'
EDIT_LIGHT = '#f6efdf'
TEXT = '#1a1a1a'

def box(x, y, w, h, text=None, fontsize=8.5, stroke=BOX_STROKE, fill=BOX_FILL,
        weight='normal', radius=2.0, text_color=TEXT):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           linewidth=1.0, edgecolor=stroke, facecolor=fill)
    ax.add_patch(patch)
    if text:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, weight=weight)

def arrow(x1, y1, x2, y2, label=None, color=BOX_STROKE, lw=1.2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=11,
                        color=color, linewidth=lw)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2 + 2.5, (y1+y2)/2, label, fontsize=7.2, color=TEXT,
                ha='left', va='center', style='italic')

# ---------------------------------------------------------------------------
# Step 1: Corpus v3.4
# ---------------------------------------------------------------------------
box(5, 91, 90, 7,
    'Corpus v3.4 — high-extraversion edit (104 atoms)',
    fontsize=9, weight='bold')

arrow(50, 90.5, 50, 86, label=r' administer Mini-IPIP')

# ---------------------------------------------------------------------------
# Step 2: WRONG direction
# ---------------------------------------------------------------------------
box(8, 76.4, 84, 8.6, '', fill=WRONG_LIGHT, stroke=WRONG)
ax.text(50, 82, r'Mini-IPIP $\Delta = -0.50$', ha='center', va='center',
        fontsize=10, weight='bold', color=WRONG)
ax.text(50, 78.5, '✗ wrong direction', ha='center', va='center',
        fontsize=8.5, color=WRONG, style='italic')

arrow(50, 76, 50, 71.5, label=r' inspect fired-atom log')

# ---------------------------------------------------------------------------
# Step 3: Audit (fired-atom log)
# ---------------------------------------------------------------------------
box(5, 50, 90, 21, '', radius=1.5)
ax.text(50, 68.5, 'Fired-atom log  (per-response audit, §A.3)',
        ha='center', va='center', fontsize=8.5, weight='bold', color=TEXT)

# 4 fired atoms, with the 2 buggy ones flagged
atoms = [
    ('highe_002', '"Others charge" → I stay quiet, defer', '✗', WRONG),
    ('highe_017', '"Others charge" → I let them lead', '✗', WRONG),
    ('highe_021', 'When invited → I show up energized', '✓', RIGHT_C),
    ('highe_055', 'At a party → I start conversations', '✓', RIGHT_C),
]
for i, (aid, body, mark, color) in enumerate(atoms):
    y = 64.5 - i * 3.2
    ax.text(8, y, aid, fontsize=7.2, family='monospace', color=TEXT, ha='left')
    ax.text(28, y, body, fontsize=7.4, color=TEXT, ha='left', family='monospace')
    ax.text(91, y, mark, fontsize=11, color=color, ha='right',
            weight='bold')

ax.text(50, 52, '↳ 2 of 104 atoms silently polarity-inverted by MiniMax-M2.7 verbaliser',
        ha='center', va='center', fontsize=7.2, color=EDIT, style='italic')

arrow(50, 49.5, 50, 45, label=r' 2-row JSONL edit', color=EDIT)

# ---------------------------------------------------------------------------
# Step 4: Edit
# ---------------------------------------------------------------------------
box(5, 31.2, 90, 13.8, '', fill=EDIT_LIGHT, stroke=EDIT, radius=1.5)
ax.text(50, 42.5, 'Targeted edit', ha='center', va='center',
        fontsize=8.5, weight='bold', color=EDIT)
ax.text(50, 39, r'"Others charge" $=$ "others take charge"',
        ha='center', va='center', fontsize=7.7, color=TEXT,
        family='monospace')
ax.text(50, 36, 'fix verbaliser misreading; flip the 2 atoms',
        ha='center', va='center', fontsize=7.7, color=TEXT)
ax.text(50, 33.5, '+ 2 worked examples added to verbalise prompt',
        ha='center', va='center', fontsize=7, color='#666', style='italic')

arrow(50, 30.9, 50, 27, label=r' re-administer (same model)')

# ---------------------------------------------------------------------------
# Step 5: Corpus v3.5
# ---------------------------------------------------------------------------
box(5, 19, 90, 7,
    'Corpus v3.5 — same edit, 2 atoms corrected',
    fontsize=9, weight='bold')

arrow(50, 18.5, 50, 14, label='')

# ---------------------------------------------------------------------------
# Step 6: RIGHT direction
# ---------------------------------------------------------------------------
box(8, 4.4, 84, 8.6, '', fill=RIGHT_LIGHT, stroke=RIGHT_C)
ax.text(50, 10, r'Mini-IPIP $\Delta = +0.58$', ha='center', va='center',
        fontsize=10, weight='bold', color=RIGHT_C)
ax.text(50, 6.5, '✓ right direction (N = 3)', ha='center', va='center',
        fontsize=8.5, color=RIGHT_C, style='italic')

# Footer note
ax.text(50, -1.5,
        'Localization to 2 atoms · structurally unavailable to\n'
        'persona-as-string and persona-as-vector approaches',
        ha='center', va='center', fontsize=7.5, color='#555555', style='italic')

ax.set_ylim(-5, 100)
plt.tight_layout()
plt.savefig('figure2_case_study.pdf', bbox_inches='tight',
            pad_inches=0.05)
plt.savefig('figure2_case_study.png', bbox_inches='tight',
            pad_inches=0.05, dpi=180)
print("Figure 2 saved.")
