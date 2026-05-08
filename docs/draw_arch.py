"""
Architecture diagram for:
  Direct Multi-Frame Action + Zero Action History
  (Variant D + E: future_action_direct_F18_2head_zero_action)

Run:  python draw_arch.py
Out:  arch_variant_D_E.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── palette ─────────────────────────────────────────────────────────
C = dict(
    blue    = '#DBEAFE', violet  = '#EDE9FE', amber   = '#FEF3C7',
    green   = '#D1FAE5', ao      = '#A7F3D0', red     = '#FEE2E2',
    gray    = '#F1F5F9', yellow  = '#FEFCE8', emerald = '#ECFDF5',
    sky     = '#E0F2FE', ink     = '#1E293B', border  = '#64748B',
    redbdr  = '#EF4444', redfill = '#FEF2F2',
)

fig, ax = plt.subplots(figsize=(26, 14))
ax.set_xlim(-0.5, 26.5)
ax.set_ylim(-2.5, 13.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── primitives ──────────────────────────────────────────────────────

def box(cx, cy, w, h, fill, lines, fs=9, bold0=True,
        bdr=None, bdrw=1.5, bdr_color=None):
    """Rounded rectangle + centred multi-line text."""
    ec = bdr_color or C['border']
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle='round,pad=0.12',
                       fc=fill, ec=ec, lw=bdrw, zorder=3, clip_on=False)
    ax.add_patch(p)
    if isinstance(lines, str):
        lines = [lines]
    lh = fs * 0.042            # line-height in data coords
    y0 = cy + (len(lines)-1)/2 * lh
    for i, ln in enumerate(lines):
        fw = 'bold' if (i == 0 and bold0) else 'normal'
        ax.text(cx, y0 - i*lh, ln, ha='center', va='center',
                fontsize=fs, color=C['ink'], fontweight=fw,
                zorder=4, clip_on=False)

def tok(cx, cy, w, h, fill, lines, fs=8.5):
    """Token box inside the sequence strip."""
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle='round,pad=0.06',
                       fc=fill, ec=C['border'], lw=1.2, zorder=5, clip_on=False)
    ax.add_patch(p)
    if isinstance(lines, str):
        lines = [lines]
    lh = 0.30
    y0 = cy + (len(lines)-1)/2 * lh
    for i, ln in enumerate(lines):
        fw = 'bold' if i == 0 else 'normal'
        ax.text(cx, y0 - i*lh, ln, ha='center', va='center',
                fontsize=fs, color=C['ink'], fontweight=fw,
                zorder=6, clip_on=False)

def arrow(x1, y1, x2, y2, lbl='', lw=1.5, color=None, rad=0.0,
          dashed=False, ls_color=None):
    clr = color or C['border']
    ls  = 'dashed' if dashed else 'solid'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=clr, lw=lw,
                                linestyle=ls,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=5, clip_on=False)
    if lbl:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.22, lbl, ha='center', va='bottom',
                fontsize=7.5, color='#64748B', style='italic', zorder=6)

def annbox(cx, cy, w, h, lines, fs=8.5):
    """Red-bordered annotation box."""
    box(cx, cy, w, h, C['redfill'], lines, fs=fs,
        bdr_color=C['redbdr'], bdrw=1.4)


# ═══════════════════════════════════════════════════════════════════
# 1. INPUTS
# ═══════════════════════════════════════════════════════════════════
box(1.5, 10.4, 2.5, 1.5, C['blue'],
    ['Video Frame', 'B×T×3×192×192', 'uint8'], fs=9)
box(1.5, 7.6, 2.5, 1.5, C['violet'],
    ['Text Instruction', '(optional, string)'], fs=9)

# ═══════════════════════════════════════════════════════════════════
# 2. ENCODERS
# ═══════════════════════════════════════════════════════════════════
box(5.1, 10.4, 3.0, 2.0, C['amber'],
    ['Image Tokenizer', 'ConvTokenizer', 'EfficientNet-B0+MLP', '→ (B,T,1,1024)'], fs=9)
box(5.1, 7.6, 3.0, 2.0, C['violet'],
    ['Text Tokenizer', 'GemmaEmbed-300M', '→(B,T,1,768)', '→MLP→(B,T,1,1024)'], fs=9)

arrow(2.75, 10.4, 3.6,  10.4)
arrow(2.75,  7.6, 3.6,   7.6)

# ═══════════════════════════════════════════════════════════════════
# 3. SEQUENCE BLOCK
# ═══════════════════════════════════════════════════════════════════
SBG_CX, SBG_CY, SBG_W, SBG_H = 10.0, 9.0, 5.0, 4.0
bg = FancyBboxPatch((SBG_CX-SBG_W/2, SBG_CY-SBG_H/2), SBG_W, SBG_H,
                    boxstyle='round,pad=0.15',
                    fc=C['gray'], ec='#CBD5E1', lw=1.2, zorder=2, clip_on=False)
ax.add_patch(bg)
ax.text(SBG_CX, SBG_CY+SBG_H/2-0.25,
        'Per-Step Token Layout  ×T steps (causal)',
        ha='center', va='top', fontsize=8.5, color='#64748B',
        style='italic', zorder=4)

# token strip at y=10.0
TOK_Y   = 10.0
TOK_H   = 0.90
GAP     = 0.08
TOKENS  = [
    ('txt', '(1)',  C['violet'],  0.78),
    ('img', '(1)',  C['amber'],   0.88),
    ('thk', '(1)',  C['green'],   0.78),
    ('a⁰', '(1)',   C['ao'],      0.78),
    ('aᵢₙ', '≡ 0', C['red'],     0.85),
]
total_w = sum(t[3] for t in TOKENS) + GAP*(len(TOKENS)-1)
xc = SBG_CX - total_w/2
tok_cx = []
for name, sub, color, w in TOKENS:
    cx = xc + w/2
    tok_cx.append(cx)
    tok(cx, TOK_Y, w, TOK_H, color, [name, sub])
    xc += w + GAP

# causal mask description
mask_text = [
    'Causal mask rules:',
    'img/txt/thk: see previous steps (cross-step)',
    'a⁰: sees img/txt/thk in current step only',
    'a⁰ cannot be used as KV (prevents leakage)',
    'aᵢₙ = 0: carries position info only',
]
yt = 8.75
for i, ln in enumerate(mask_text):
    fs = 8.8 if i == 0 else 8.0
    fw = 'bold' if i == 0 else 'normal'
    ax.text(SBG_CX, yt - i*0.42, ln,
            ha='center', va='center', fontsize=fs,
            color='#475569', fontweight=fw, zorder=4)

# arrows: encoders → sequence block
arrow(6.6, 10.4, tok_cx[0], TOK_Y+TOK_H/2, rad=-0.15)
arrow(6.6,  7.6, tok_cx[0], TOK_Y-TOK_H/2, rad= 0.30)

# zero_action annotation
annbox(SBG_CX, 5.55, 5.2, 1.55,
       ['zero_action_input = True',
        'Train: aᵢₙ ← pos-tokens only  (no GT action embedding)',
        'Infer: aᵢₙ = 0,  skip Pass 2  (single forward pass)',
        '→ Eliminates teacher-forcing / AR inference mismatch'])

arrow(tok_cx[4], TOK_Y-TOK_H/2-0.05,
      tok_cx[4], 5.55+0.82,
      dashed=True, color=C['redbdr'], lw=1.4, rad=0.0)

# ═══════════════════════════════════════════════════════════════════
# 4. POLICY CAUSAL TRANSFORMER
# ═══════════════════════════════════════════════════════════════════
TFM_CX, TFM_CY = 14.8, 9.0
box(TFM_CX, TFM_CY, 3.0, 4.0, C['gray'],
    ['Policy Causal Transformer', '',
     '• 10 layers',
     '• d = 1024',
     '• 16 heads (MHA)',
     '• RoPE',
     '• FlexAttention block_mask',
     '', '', ''], fs=9)
# red warning text inside transformer
ax.text(TFM_CX, TFM_CY-1.35,
        'skip_action_decoder = True', ha='center', va='center',
        fontsize=8.5, color='#DC2626', fontweight='bold', zorder=5)
ax.text(TFM_CX, TFM_CY-1.72,
        '(ActionDecoder not used)', ha='center', va='center',
        fontsize=8.0, color='#DC2626', style='italic', zorder=5)

arrow(SBG_CX+SBG_W/2, SBG_CY, TFM_CX-1.5, TFM_CY,
      lbl='(B, T·L, 1024)')

# ═══════════════════════════════════════════════════════════════════
# 5a. PICK a⁰ OUTPUT
# ═══════════════════════════════════════════════════════════════════
AO_CX, AO_CY = 18.3, 10.8
box(AO_CX, AO_CY, 2.8, 1.35, C['green'],
    ['Pick a⁰ output position',
     'offset: n_img + n_txt + n_thk',
     '→ (B, T, 1024)'], fs=8.8)
arrow(TFM_CX+1.5, TFM_CY, AO_CX-1.4, AO_CY, lbl='full output')

# ═══════════════════════════════════════════════════════════════════
# 5b. direct_action_mlp
# ═══════════════════════════════════════════════════════════════════
MLP_CX, MLP_CY = 18.3, 8.7
box(MLP_CX, MLP_CY, 2.8, 1.65, C['yellow'],
    ['direct_action_mlp',
     'Linear(1024→512) + SiLU',
     'Linear(512→256)  + SiLU',
     '→ (B, T, 256)'], fs=8.8)
arrow(AO_CX, AO_CY-0.70, MLP_CX, MLP_CY+0.87)

# ═══════════════════════════════════════════════════════════════════
# 6. TWO PARALLEL HEADS
# ═══════════════════════════════════════════════════════════════════
BTN_CX, BTN_CY = 22.5, 11.0
STK_CX, STK_CY = 22.5,  7.8

box(BTN_CX, BTN_CY, 3.8, 2.1, C['emerald'],
    ['direct_button_head',
     'Linear(256 → 18×12 = 216)',
     'reshape → (B, T, F=18, n_b=12)',
     'Loss: BCEWithLogitsLoss',
     '12 Cuphead buttons, multi-label',
     'Infer: σ(·) > 0.5,  use f = 0'], fs=8.8)

box(STK_CX, STK_CY, 3.8, 2.1, C['sky'],
    ['direct_stick_head',
     'Linear(256 → 18×4×3 = 216)',
     'reshape → (B, T, F=18, 4, S=3)',
     'Loss: 4 × CrossEntropy',
     'LX / LY / RX / RY,  3 bins each',
     'Infer: argmax,  use f = 0'], fs=8.8)

arrow(MLP_CX+1.4, MLP_CY, BTN_CX-1.9, BTN_CY, rad=-0.22)
arrow(MLP_CX+1.4, MLP_CY, STK_CX-1.9, STK_CY, rad= 0.22)

# "2-HEAD" bracket
bx = BTN_CX - 2.05
ax.annotate('', xy=(bx, STK_CY+0.5), xytext=(bx, BTN_CY-0.5),
            arrowprops=dict(arrowstyle='-[', color='#94A3B8',
                            lw=1.8, mutation_scale=14), zorder=4)
ax.text(bx-0.38, (BTN_CY+STK_CY)/2, '2-HEAD',
        ha='center', va='center', fontsize=8.5,
        color='#94A3B8', fontweight='bold', rotation=90)

# output shapes
ax.text(24.55, BTN_CY,
        'buttons:\n(B, T, F=18, 12)\n← 18 future frames',
        ha='left', va='center', fontsize=8.0, color='#475569', zorder=4)
ax.text(24.55, STK_CY,
        'LX, LY, RX, RY:\neach (B, T, F=18, S=3)\n← S=3 bins',
        ha='left', va='center', fontsize=8.0, color='#475569', zorder=4)

# ═══════════════════════════════════════════════════════════════════
# 7. INFERENCE / TRAINING NOTE
# ═══════════════════════════════════════════════════════════════════
INF_CX, INF_CY = 22.0, 5.4
annbox(INF_CX, INF_CY, 5.6, 1.6,
       ['Inference:  use frame f = 0 only',
        'b̂ = σ(buttons[…, f=0, :]) > 0.5       (12 buttons, multi-label)',
        'ŝ = argmax(stick[…, f=0, :], dim=-1)   (LX/LY/RX/RY)',
        'Training:  supervise f = 0, 1, …, 17   (F=18 future frames)'])
arrow(STK_CX, STK_CY-1.1, INF_CX, INF_CY+0.85,
      dashed=True, color=C['redbdr'], lw=1.2, rad=0.15)

# ═══════════════════════════════════════════════════════════════════
# 8. TITLE
# ═══════════════════════════════════════════════════════════════════
ax.text(12.5, 13.1,
        'Direct Multi-Frame Action Prediction + Zero Action History',
        ha='center', va='center', fontsize=16,
        color=C['ink'], fontweight='bold')
ax.text(12.5, 12.55,
        'future_action_direct_F18_2head_zero_action  (Variant D + E)',
        ha='center', va='center', fontsize=11,
        color='#64748B', style='italic')

# ═══════════════════════════════════════════════════════════════════
# 9. LEGEND
# ═══════════════════════════════════════════════════════════════════
legend_items = [
    ('txt',      C['violet']),
    ('img',      C['amber']),
    ('thk',      C['green']),
    ('a⁰',       C['ao']),
    ('aᵢₙ ≡ 0', C['red']),
]
ax.text(0.2, -1.65, 'Token types:', ha='left', va='center',
        fontsize=8.5, color='#475569', fontweight='bold')
lx = 1.8
for label, color in legend_items:
    p = FancyBboxPatch((lx, -1.95), 0.55, 0.42,
                       boxstyle='round,pad=0.04',
                       fc=color, ec=C['border'], lw=1.0,
                       zorder=3, clip_on=False)
    ax.add_patch(p)
    ax.text(lx+0.55+0.12, -1.74, label,
            ha='left', va='center', fontsize=8.5,
            color=C['ink'], zorder=4)
    lx += len(label)*0.22 + 1.1

# ───────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
out = '/home/ch/open-p2p_stamo/nitrogen-gamepad-tcp/docs/arch_variant_D_E.png'
plt.savefig(out, dpi=160, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f'Saved: {out}')
