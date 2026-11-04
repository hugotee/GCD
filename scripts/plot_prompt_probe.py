#!/usr/bin/env python3
"""Plot measured raw-query prompt effects and text direction changes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

p = argparse.ArgumentParser()
p.add_argument('--summary', type=Path, required=True)
p.add_argument('--font', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
font_manager.fontManager.addfont(str(a.font))
name = font_manager.FontProperties(fname=str(a.font)).get_name()
assert name == 'Times New Roman'
plt.rcParams.update({'font.family': name, 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none'})
j = json.loads(a.summary.read_text())
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
labels = ['B1 / old15', 'B1 / all20', 'B2 / old15', 'B2 / all20']
values = [j['conditions'][s.replace(' / ', '_')]['gt_score_coverage_percent']['0.3'] for s in labels]
axes[0].bar(range(4), values, color=['#238973', '#238973', '#c34a36', '#c34a36'], width=.65)
for i, value in enumerate(values):
    axes[0].text(i, value + 2, f'{value:.1f}%', ha='center')
axes[0].set(xticks=range(4), xticklabels=labels, ylim=(0, 108),
            ylabel='Old GT with true-class score >= 0.3 (%)',
            title='Old-class scores under fixed prompts')
geometry = j['B1_to_B2_text_geometry']['all20']
differences = [max(0, geometry['pre']['mean_one_minus_cosine']),
               geometry['fused']['mean_one_minus_cosine'],
               j['same_model_prompt_text_geometry']['B2']['mean_one_minus_cosine']]
axes[1].bar(range(3), differences, color=['#737373', '#175d91', '#8f75a8'], width=.6)
for i, value in enumerate(differences):
    axes[1].text(i, value + .002, f'{value:.4f}', ha='center')
axes[1].set(xticks=range(3), xticklabels=['B1 -> B2\npre-encoder', 'B1 -> B2\nfused text', 'B2: old15 -> all20\nfused text'],
            ylim=(0, .09), ylabel='Mean 1 - cosine of corresponding class vectors',
            title='Text direction changes after the encoder')
for ax in axes:
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
fig.text(.5, .015, '128 trainval images; 327 old GT; raw900 queries. Candidate coverage is not detection AP or causal identification.',
         ha='center', fontsize=10)
fig.subplots_adjust(left=.07, right=.99, bottom=.20, top=.86, wspace=.28)
a.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=200)
fig.savefig(a.out.with_suffix('.svg'))
