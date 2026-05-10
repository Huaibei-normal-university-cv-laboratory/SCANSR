import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
# 数据准备
k_labels = ['[1/2]', '[1/8,1/5]', '[1/7,1/4]', '[1/6,1/3]', '[1/5,1/2]', '[1/4,2/3]',
            '[1/3,3/4]', '[1/2,4/5]', '[2/3,5/6]']
psnr = [37.0486,
36.9996,
37.02,
37.063,
37.1013,
37.1165,
37.1214,
37.1465,
37.1067,
]

# 创建画布
fig, ax1 = plt.subplots(figsize=(10, 5))

# 绘制PSNR（k1红色，其他蓝色）
ax1.plot(k_labels[0:1], psnr[0:1], 'r-o', markersize=6)  # 红色k1
ax1.plot(k_labels[1:], psnr[1:], 'b-o', markersize=6)    # 蓝色k2-k6
ax1.set_ylabel('PSNR(dB)')
ax1.set_ylim(36.9, 37.2)
ax1.tick_params(axis='y')
# 高亮最优值
ax1.plot(7, psnr[7], 'r-o', markersize=8)

# 添加横坐标说明
ax1.set_xlabel('k', labelpad=10)

# 标题和网格
# plt.title("PSNR Trends with Different k Values", pad=20)
ax1.grid(linestyle='--', alpha=0.6)

# 关闭网格
ax1.grid(False)

fig.tight_layout()
plt.show()