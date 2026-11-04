#!/usr/bin/env python3
"""Scientific diagnostic plot using a verified, real Times New Roman font."""
import argparse
import csv
import json
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--font', type=Path, required=True)
a = p.parse_args()
if not a.font.is_file():
    raise FileNotFoundError('A real Times New Roman font file is required')
font_manager.fontManager.addfont(str(a.font))
family = font_manager.FontProperties(fname=str(a.font)).get_name()
assert family == 'Times New Roman', family
plt.rcParams.update({'font.family': family, 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none'})
summary = json.loads((a.input / 'summary.json').read_text())
rows = list(csv.DictReader((a.input / 'task2_hidden_old_exposure.csv').open()))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
x = [100 * float(r['hidden_fraction_of_trainval_instances']) for r in rows]
y = [float(r['B1_minus_B2_VOC07_AP50']) for r in rows]
axes[0].scatter(x, y, color='#175d91', s=38, alpha=.85)
offsets = {'person': (8, 0), 'chair': (-40, 7), 'diningtable': (7, 3), 'bottle': (7, -2), 'motorbike': (8, 2)}
for row, xv, yv in zip(rows, x, y):
    if row['category'] in offsets:
        axes[0].annotate(row['category'], (xv, yv), xytext=offsets[row['category']], textcoords='offset points')
axes[0].set(xlabel='Old instances hidden in Task2 (%)', ylabel='B1 - B2 VOC07 AP50 (points)',
            title='Training exposure and old-class AP loss', xlim=(-2, 40), ylim=(-3, 85))
axes[0].text(.03, .95, 'Descriptive association: r = %.3f (15 classes)' %
             summary['exposure_forgetting_pearson_over_15_classes'], transform=axes[0].transAxes, va='top', fontsize=10)
for model, color, marker in [('B0', '#737373', 's'), ('B1', '#188671', 'o'), ('B2', '#c34a36', '^')]:
    coverage = summary['models'][model]['old_gt_coverage_by_threshold']
    thresholds = sorted(float(v) for v in coverage)
    values = [100 * coverage[str(v)]['same_class_iou50'] / coverage[str(v)]['old_gt'] for v in thresholds]
    axes[1].plot(thresholds, values, color=color, marker=marker, linewidth=1.8, label=model)
axes[1].set(xlabel='Score threshold', ylabel='Old GT with same-class IoU > 0.5 candidate (%)',
            title='Cached old-class GT coverage', xlim=(-.02, .52), ylim=(0, 105))
axes[1].legend(frameon=False, loc='lower left')
for ax in axes:
    ax.grid(axis='y', alpha=.18)
fig.text(.5, .015, 'Top300 cached detections only. Candidate coverage is not official recall. Association is not causation.',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0, .055, 1, 1))
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=200)
fig.savefig(a.out.with_suffix('.svg'))
a.out.with_suffix('.font.json').write_text(json.dumps(dict(font_family=family, font_file=str(a.font),
    font_sha256=hashlib.sha256(a.font.read_bytes()).hexdigest(), matplotlib_version=matplotlib.__version__), indent=2) + '\n')
