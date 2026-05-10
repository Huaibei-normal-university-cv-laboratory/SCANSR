import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np

# 数据
datasets = ['Rain1400', 'Rain1200', 'Rain200L']
psnr_wok = [34.02, 34.78, 41.02]
psnr_w   = [34.33, 35.04, 41.23]
y = np.arange(len(datasets))

# x 轴断开的三个区间
xlims = ((34.0, 34.8), (35.0, 41.0), (41.0, 41.3))

fig = plt.figure(figsize=(7, 3))
bax = brokenaxes(xlims=xlims, hspace=0.05, despine=False)

bar_height = 0.4
bax.barh(y - bar_height/2, psnr_wok, height=bar_height, color='blue',  label='w/o Top-k')
bax.barh(y + bar_height/2, psnr_w,   height=bar_height, color='red',   label='w Top-k')

def annotate_on_broken(xval, yval, text, ha, va):
    for idx, (xmin, xmax) in enumerate(xlims):
        if xmin <= xval <= xmax:
            ax = bax.axs[idx]      # ← 只要这一行从 [0][idx] 改成 [idx]
            ax.text(xval, yval, text, ha=ha, va=va, fontsize=9)
            break

for i, (v0, v1) in enumerate(zip(psnr_wok, psnr_w)):
    annotate_on_broken(v0 - 0.02, i - bar_height/2, f'{v0:.2f}', ha='right', va='center')
    annotate_on_broken(v1 + 0.02, i + bar_height/2, f'{v1:.2f}', ha='left',  va='center')

bax.set_yticks(y)
bax.set_yticklabels(datasets, fontsize=10)
bax.set_xlabel('PSNR (dB)', fontsize=11)
bax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()
