import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（避免中文乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ------------- 1. 数据准备 -------------
# 指标名称（按展示顺序）
metrics = [
    '语义内聚性（SC）',
    '服务耦合度（SCP）',
    '服务边界清晰度（SBC）',
    '内部服务依赖密度（ISDD）',
    '服务依赖熵（SDE）',
    '服务规模平衡性（SSB）',
    'Nano服务占比（R_nano）',
    '结构不稳定性指数（SII）',
    '内部调用占比（ICP）',
    '模块度（Modularity）',
    '服务循环依赖比例（SCDR）'
]

# 两个划分结果的数值
constraint_solve = [0.8159, 0.7880, 0.5177, 0.1408, 0.6138, 0.6867, 0.0000, 0.4969, 0.7616, 0.3900, 0.4286]
mono2micro = [0.8130, 0.7823, 0.5172, 0.1471, 0.1017, 0.7565, 0.2857, 0.6200, 0.6954, 0.3579, 0.5714]

# 指标优劣方向：higher_better=True 越高越好，False 越低越好
higher_better = [
    True,   # SC 越高越好
    False,  # SCP 越低越好
    True,   # SBC 越高越好
    True,   # ISDD 越高越好
    False,  # SDE 越低越好
    False,  # SSB 越低越好
    False,  # R_nano 越低越好
    False,  # SII 越低越好
    True,   # ICP 越高越好
    True,   # Modularity 越高越好
    False   # SCDR 越低越好
]

# ------------- 2. 自动判断每个指标的更优值 -------------
def get_better_index(value1, value2, is_higher_better):
    """
    判断两个数值中哪个更优，返回更优值的索引（0=constraint_solve，1=mono2micro，-1=相等）
    """
    if value1 == value2:
        return -1
    if is_higher_better:
        return 0 if value1 > value2 else 1
    else:
        return 0 if value1 < value2 else 1

# 预计算每个指标的更优方案索引
better_indices = []
for v1, v2, hb in zip(constraint_solve, mono2micro, higher_better):
    better_indices.append(get_better_index(v1, v2, hb))

# ------------- 3. 可视化配置 -------------
x = np.arange(len(metrics))  # 指标x轴位置
width = 0.35  # 柱状图宽度
colors = ['#1f77b4', '#ff7f0e']  # 两个方案的颜色
better_label_color = 'darkgreen'  # “更好”标注的颜色
better_label_fontsize = 12  # 【修改1】放大“更好”标注字体（原10）
better_label_fontsize = 12  # “更好”标注的字体大小

# 创建画布（调整大小以适应更多指标）
fig, ax = plt.subplots(figsize=(18, 9))

# 绘制柱状图
rects1 = ax.bar(x - width / 2, constraint_solve, width, label='课题方案', color=colors[0])
rects2 = ax.bar(x + width / 2, mono2micro, width, label='mono2micro', color=colors[1])

# ------------- 4. 图表美化 & 标注“更好” -------------
# 添加标题和轴标签 - 【修改2】放大标题和轴标签字体
ax.set_title('微服务划分结果指标对比', fontsize=22, pad=20)  # 原18
ax.set_ylabel('指标数值', fontsize=16)  # 原14
ax.set_xlabel('评估指标', fontsize=16)  # 原14

# 设置x轴刻度和标签 - 【修改3】放大x轴标签字体
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=11)  # 调整角度以适应更多指标

# 添加优劣方向标注 - 【修改4】放大优劣方向标注字体
for i, (metric, hb) in enumerate(zip(metrics, higher_better)):
    direction = '↑ 越高越好' if hb else '↓ 越低越好'
    ax.text(i, ax.get_ylim()[1] * 1.03, direction, ha='center', va='bottom',
            fontsize=12, color='darkred', fontweight='bold')  # 原10

# 添加数值标签 + 标注“更好”
def add_value_and_better_labels(rects, rects_other, indices, is_first_group):
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        # 1. 添加数值标签 - 【修改5】放大数值标签字体
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 数值标签偏移量
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)  # 原9

        # 2. 判断是否是当前指标的更优方案，若是则标注“更好”
        better_idx = indices[idx]
        if (better_idx == 0 and is_first_group) or (better_idx == 1 and not is_first_group):
            ax.annotate('更好',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 15),  # “更好”标签偏移量
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=better_label_fontsize,
                        color=better_label_color, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.7))

# 为两个分组添加标签（约束求解=第一个分组，mono2micro=第二个分组）
add_value_and_better_labels(rects1, rects2, better_indices, is_first_group=True)
add_value_and_better_labels(rects2, rects1, better_indices, is_first_group=False)

# 添加图例 - 【修改6】放大图例字体
ax.legend(loc='upper right', fontsize=14)  # 原12

# 调整y轴上限（避免“更好”标签超出画布）
y_max = max(max(constraint_solve), max(mono2micro)) * 1.2
ax.set_ylim(0, y_max)

# 调整布局（防止标签被截断）
plt.tight_layout()

# ------------- 5. 保存/显示图表 -------------
plt.savefig('微服务划分对比图.png', dpi=300, bbox_inches='tight')  # 保存高清图片
plt.show()