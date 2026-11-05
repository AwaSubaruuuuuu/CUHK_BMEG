'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

# 1. Generate dataset
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
X = X.flatten()
y = y.flatten()
N = len(X)

# 2. Starting values
intercept = 2.0
slope = 3.0
learning_rate = 0.05
n_iterations = 100

# List to record error history for plotting
mse_history = []

print("=========================================================")
print(f"{'Iteration':<10} | {'Slope':<10} | {'Intercept':<10} | {'Error (MSE)':<15}")
print("=========================================================")

# 3. Gradient Descent Loop
for i in range(n_iterations):

  # Step A: Predict & Calculate Error
  y_pred = slope * X + intercept
  error = y_pred - y
  mse = np.mean(error**2)
  mse_history.append(mse)

  # Print early steps (1-5), every 10 steps, and final step to highlight the drop
  if i < 5 or (i + 1) % 10 == 0 or i == n_iterations - 1:
    print(f"{i+1:<10d} | {slope:<10.4f} | {intercept:<10.4f} | {mse:<15.4f}")

  # Step B: Calculate Gradients
  grad_intercept = (2 / N) * np.sum(error)
  grad_slope = (2 / N) * np.sum(error * X)

  # Step C: Update Parameters
  intercept -= learning_rate * grad_intercept
  slope -= learning_rate * grad_slope

print("=========================================================\n")


# 4. Side-by-Side Visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Graph 1: Error Curve (Big to Small)
ax1.plot(range(1, n_iterations + 1), mse_history, color="purple", linewidth=2.5)
ax1.set_xlabel("Iteration Step")
ax1.set_ylabel("Mean Squared Error (MSE)")
ax1.set_title("1. Error Decreasing Over Time (Big → Small)")
ax1.grid(True, linestyle="--", alpha=0.5)

# Graph 2: Final Line Fit
ax2.scatter(X, y, color="blue", alpha=0.7, label="Actual Data")
ax2.plot(
    X,
    slope * X + intercept,
    color="red",
    linewidth=2.5,
    label=f"Fitted Line (y = {slope:.2f}x + {intercept:.2f})",
)
ax2.set_xlabel("Feature (X)")
ax2.set_ylabel("Target (y)")
ax2.set_title("2. Optimized Regression Line")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
'''

#Decision Tree on Breast Cancer Dataset

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import math

import warnings
warnings.filterwarnings('ignore')

#load Dataset
data=pd.read_csv("Breast_Cancer_Dataset.csv")

data.info()

data.head()

print("Shape:", data.shape)
print(data.isnull().sum())

#preprocessing
df=data.drop(['id'], axis=1)

# 基于乳腺癌数据集的决策树分类
# 将目标列 'diagnosis' 中的字符串标签转换为数值：'M' (恶性) 映射为 1，'B' (良性) 映射为 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 分离特征（X）和目标变量（y）
X = df.drop('diagnosis', axis=1) # 删除标签列，保留所有特征作为输入 X
y = df['diagnosis']              # 提取标签列作为预测目标 y

X.head() # 查看特征数据的前几行
y.head() # 查看目标数据的前几行

# 划分训练集和测试集
# test_size=0.2 表示 20% 的数据用于测试，80% 用于训练；random_state=42 确保每次运行划分结果一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 模型训练
# 实例化决策树分类器，指定分裂标准为 'entropy'（信息熵）
model = DecisionTreeClassifier(criterion='entropy')
# 使用训练集数据拟合（训练）模型
model.fit(X_train, y_train)


# ---- 纯手工实现的信息熵与信息增益函数（用于理解底层原理） ----

# 计算信息熵 (Entropy)
def entropy(column):
    counts = column.value_counts()           # 统计每个类别出现的次数
    probabilities = counts / len(column)     # 计算每个类别的概率
    # 应用香农熵公式: -sum(p * log2(p))
    return -sum(probabilities * probabilities.apply(math.log2))

# 计算条件熵 (Conditional Entropy)
def conditional_entropy(data, feature, target):
    values = data[feature].unique()          # 获取该特征列的所有唯一值
    weighted_entropy = 0                     # 初始化加权熵

    for value in values:
        # 筛选出该特征等于当前值的数据子集
        subset = data[data[feature] == value]
        # 累加：(子集样本数 / 总样本数) * 子集的标签熵
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target])

    return weighted_entropy

# 计算信息增益 (Information Gain)
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])                    # 计算划分前的总信息熵
    cond_entropy = conditional_entropy(data, feature, target)# 计算划分后的条件熵
    return total_entropy - cond_entropy                      # 信息增益 = 总熵 - 条件熵

# -------------------------------------------------------------

# 打印每个特征的信息增益
print("\n每个特征的信息增益 (Information Gain for each feature):\n")
for feature in X.columns:
    ig = information_gain(df, feature, 'diagnosis')
    print(f"{feature}: {ig}")

# 预测与模型评估
y_pred = model.predict(X_test) # 使用测试集特征进行预测

# 计算预测准确率
accuracy = accuracy_score(y_test, y_pred) * 100
print("\n准确率 (Accuracy):", accuracy)

# 打印分类报告（包含精确率 Precision、召回率 Recall、F1-Score 等详细指标）
print("\n分类报告 (Classification Report):\n")
print(classification_report(y_test, y_pred))

# 决策树可视化
plt.figure(figsize=(12, 8)) # 设置画布大小
plot_tree(
    model,
    filled=True,                        # 节点填充颜色，颜色深浅表示纯度
    feature_names=X.columns,            # 节点中显示的特征名称
    class_names=['Benign', 'Malignant'] # 类别名称：0对应Benign(良性)，1对应Malignant(恶性)
)
plt.title("决策树可视化 (Decision Tree)")
plt.show() # 展示图像

# 新样本预测（输入一个未知肿瘤的 30 个特征指标）
new_sample = [[
    12.5, 19.2, 80.0, 500.0, 0.085, 0.1, 0.05, 0.02, 0.17, 0.06,
    0.4, 1.0, 2.5, 40.0, 0.006, 0.02, 0.03, 0.01, 0.02, 0.003,
    16.0, 25.0, 105.0, 900.0, 0.13, 0.25, 0.28, 0.12, 0.29, 0.08
]]

# 将新样本转换为 DataFrame 格式，并带上特征列名以匹配模型输入
new_df = pd.DataFrame(new_sample, columns=X.columns)

# 进行预测
prediction = model.predict(new_df)

# 根据预测结果输出判定
if prediction[0] == 0:
    print("\n预测结果: 良性 (Benign)")
else:
    print("\n预测结果: 恶性 (Malignant)")
