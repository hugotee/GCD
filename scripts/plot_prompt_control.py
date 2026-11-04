#!/usr/bin/env python3
"""Standalone scientific figure for measured one-epoch prompt controls."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--summary', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--font', type=Path, required=True)
a = p.parse_args()
font_manager.fontManager.addfont(str(a.font))
family = font_manager.FontProperties(fname=a.font).get_name()
assert family == 'Times New Roman', family
plt.rcParams.update({'font.family': family, 'font.size': 12, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})
s = json.loads(a.summary.read_text())
assert s['passed'] and s['paired_inputs']['passed']
groups = ['old15', 'new5', 'all20']
models = [('B1 initial', s['B1_initial_groups']),
          ('new5 train prompt', s['arms']['new5']['groups']),
          ('all20 train prompt', s['arms']['all20']['groups'])]
fig, ax = plt.subplots(figsize=(8.8, 4.8))
for index, ((label, values), color) in enumerate(zip(models, ['#747c84', '#218c74', '#c14d36'])):
    bars = ax.bar(np.arange(3) + (index - 1) * .25,
                  [values[g]['VOC07_AP50'] for g in groups], .23, label=label, color=color)
    ax.bar_label(bars, fmt='%.1f', fontsize=10, padding=3)
ax.set_xticks(np.arange(3), ['Old 15 classes', 'New 5 classes', 'All 20 classes'])
ax.set_ylabel('VOC2007 AP50 (%)')
ax.set_ylim(0, 105)
fig.suptitle('Task2 training-prompt control: one complete epoch', y=.97)
ax.set_axisbelow(True); ax.grid(axis='y', alpha=.2)
ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=3, frameon=False, fontsize=10)
fig.subplots_adjust(left=.10, right=.98, bottom=.20, top=.80)
fig.text(.5, .045, 'Same B1, 1006 training images, 252 updates; 4952 test images, all20 evaluation prompt.\n'
         'Single seed. Prompt changes also affect conditioning and matching.', ha='center', fontsize=10)
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=220)
svg = a.out.with_suffix('.svg'); fig.savefig(svg)
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
a.out.with_suffix('.font.json').write_text(json.dumps(dict(font_family=family,
    font_file=str(a.font), font_sha256=hashlib.sha256(a.font.read_bytes()).hexdigest(),
    matplotlib_version=matplotlib.__version__, summary_sha256=hashlib.sha256(a.summary.read_bytes()).hexdigest()), indent=2)+'\n')
print(a.out)
