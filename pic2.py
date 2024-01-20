import pandas as pd
import matplotlib.pyplot as plt

# 从Excel读取数据
df = pd.read_excel('E:/python_lab/DMNM/result2.xlsx')  # 替换 'your_excel_file.xlsx' 为您的Excel文件路径

# 绘制折线图
plt.plot(df['Function1 Values'], df['Function2 Values'], marker='o', linestyle='-', color='black')

# 添加每个点的坐标标注
for i, row in df.iloc[:-1].iterrows():
    plt.annotate(f"({row['Function1 Values']:.8f}, {row['Function2 Values']:.8f})",
                 (row['Function1 Values'], row['Function2 Values']),
                 textcoords="offset points",
                 xytext=(2, -10),  # 调整这里的值以改变标签位置
                 ha='left')

# 添加最后一个点的坐标标注
last_row = df.iloc[-1]
plt.annotate(f"({last_row['Function1 Values']:.8f}, {last_row['Function2 Values']:.8f})",
             (last_row['Function1 Values'], last_row['Function2 Values']),
             textcoords="offset points",
             xytext=(-5, -25),  # 调整这里的值以改变标签位置
             ha='left')

# 显示图形
plt.show()
