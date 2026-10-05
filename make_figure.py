"""Render the README figure from the arrays rul_pipeline.py saves to preds.npz."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, SERIES = '#fcfcfb', '#0b0b0b', '#52514e', '#2a78d6'
Q = 25.2   # conformal half-width reported by the run in results/run.log

d = np.load('results/preds.npz')
pred, true = d['test_pred'], d['test_true']
err = pred - true
rmse = np.sqrt(np.mean(err ** 2))
cover = np.mean(np.abs(err) <= Q) * 100

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE,
                       gridspec_kw={'width_ratios': [1.25, 1]})
for a in ax:
    a.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        a.spines[s].set_color(MUTED); a.spines[s].set_linewidth(0.8)
    a.tick_params(colors=MUTED, labelsize=9)

# --- predicted vs actual, with the conformal band --------------------------
lim = [0, 130]
ax[0].fill_between(lim, [l - Q for l in lim], [l + Q for l in lim],
                   color=SERIES, alpha=0.10, linewidth=0, zorder=1)
ax[0].plot(lim, lim, color=MUTED, linewidth=2, zorder=2)
ax[0].scatter(true, pred, s=34, color=SERIES, edgecolor=SURFACE,
              linewidth=1.2, zorder=3)
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('True RUL (cycles)', color=MUTED, fontsize=10)
ax[0].set_ylabel('Predicted RUL (cycles)', color=MUTED, fontsize=10)
ax[0].set_title(f'Predicted vs. true RUL — 100 held-out engines',
                color=INK, fontsize=11.5, loc='left', pad=12)
ax[0].annotate(f'±{Q:.0f}-cycle conformal band\n{cover:.0f}% of engines inside',
               xy=(0.03, 0.95), xycoords='axes fraction', va='top',
               color=MUTED, fontsize=9)
ax[0].annotate('perfect prediction', xy=(104, 104), xytext=(60, 122),
               color=MUTED, fontsize=9,
               arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.8))

# --- error distribution ----------------------------------------------------
ax[1].axvspan(-Q, Q, color=SERIES, alpha=0.10, linewidth=0)
ax[1].hist(err, bins=np.arange(-60, 62, 6), color=SERIES,
           edgecolor=SURFACE, linewidth=1.2)
ax[1].axvline(0, color=MUTED, linewidth=2)
ax[1].set_xlabel('Prediction error (cycles)', color=MUTED, fontsize=10)
ax[1].set_ylabel('Engines', color=MUTED, fontsize=10)
ax[1].set_title(f'Error distribution — RMSE {rmse:.1f} cycles',
                color=INK, fontsize=11.5, loc='left', pad=12)
ax[1].annotate('under-predicts life\nmaintenance early (safe)', xy=(0.03, 0.96),
               xycoords='axes fraction', va='top', color=MUTED, fontsize=8.5)
ax[1].annotate('over-predicts life\nmaintenance late (risky)', xy=(0.97, 0.96),
               ha='right', xycoords='axes fraction', va='top',
               color=MUTED, fontsize=8.5)

fig.tight_layout()
fig.savefig('results/predicted_vs_actual.png', dpi=160, facecolor=SURFACE)
print(f'wrote results/predicted_vs_actual.png   RMSE {rmse:.1f}  coverage {cover:.0f}%')
