import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

# =========================================================
# 1. 自动选择可用的中文字体（关键修复）
# =========================================================
def get_available_chinese_font():
    preferred_fonts = [
        'Microsoft YaHei',        # Windows
        'SimHei',                 # Windows / Linux
        'PingFang SC',            # macOS
        'Hiragino Sans GB',       # macOS
        'WenQuanYi Micro Hei',    # Linux
        'Noto Sans CJK SC'        # Linux / Server
    ]

    available_fonts = {f.name: f.fname for f in fontManager.ttflist}

    for name in preferred_fonts:
        if name in available_fonts:
            return FontProperties(fname=available_fonts[name])

    raise RuntimeError(
        '未检测到可用的中文字体，请安装 Microsoft YaHei / SimHei / Noto Sans CJK'
    )

cn_font = get_available_chinese_font()
cn_bold = FontProperties(fname=cn_font.get_file(), weight='bold')
cn_normal = FontProperties(fname=cn_font.get_file(), weight='regular')

# 数字字体（matplotlib 内置，绝对安全）
num_bold = FontProperties(family='DejaVu Sans', weight='bold')
num_normal = FontProperties(family='DejaVu Sans', weight='regular')

plt.rcParams["axes.unicode_minus"] = False

# =========================================================
# 2. 数据
# =========================================================
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

constraint_solve = [0.8159, 0.7880, 0.5177, 0.1408, 0.6138, 0.6867, 0.0000, 0.4969, 0.7616, 0.3900, 0.4286]
mono2micro = [0.8130, 0.7823, 0.5172, 0.1471, 0.1017, 0.7565, 0.2857, 0.6200, 0.6954, 0.3579, 0.5714]
higher_better = [True, False, True, True, False, False, False, False, True, True, False]

# =========================================================
# 3. 更优指标判断
# =========================================================
def better_idx(v1, v2, higher_is_better):
    if abs(v1 - v2) < 1e-8:
        return -1
    return 0 if (v1 > v2 if higher_is_better else v1 < v2) else 1

better_indices = [
    better_idx(v1, v2, hb)
    for v1, v2, hb in zip(constraint_solve, mono2micro, higher_better)
]

# =========================================================
# 4. 表格
# =========================================================
fig, ax = plt.subplots(figsize=(22, 5))  # 调整宽度以适应更多列
ax.axis('off')

col_labels = [
    f'{m}\n({"↑" if hb else "↓"})'
    for m, hb in zip(metrics, higher_better)
]

row_labels = ['课题方案', 'mono2micro']

cell_text = [
    [f'{v:.4f}' for v in constraint_solve],
    [f'{v:.4f}' for v in mono2micro]
]

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    rowLabels=row_labels,
    cellLoc='center',
    rowLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1],
    edges='closed'
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# =========================================================
# 5. 样式（最终版）
# =========================================================
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(0.8)
    cell.set_facecolor('white')

    text = cell.get_text()

    if row == 0:  # 表头
        text.set_fontproperties(cn_bold)
        text.set_fontsize(12)

    elif col == -1:  # 行标签
        text.set_fontproperties(cn_bold)
        text.set_fontsize(12)

    elif row >= 1 and col >= 0:  # 数值
        idx = better_indices[col]
        if (row == 1 and idx == 0) or (row == 2 and idx == 1):
            text.set_fontproperties(num_bold)
        else:
            text.set_fontproperties(num_normal)
        text.set_fontsize(11)

# =========================================================
# 6. 标题与注释
# =========================================================
plt.title(
    '微服务划分结果指标对比',
    fontproperties=cn_bold,   # ★ 显式指定中文字体
    fontsize=15,
    pad=20
)

plt.figtext(
    0.5, 0.02,
    '注：↑表示指标值越高越好，↓表示指标值越低越好；加粗数值为对应指标下的最优结果',
    ha='center',
    fontproperties=cn_normal,  # ★ 显式指定中文字体
    fontsize=10,
    style='italic'
)


plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('微服务划分对比表格.png', dpi=300, bbox_inches='tight')
plt.show()
