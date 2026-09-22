# 1.导入需要的库

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# 2.设置随机种子

SEED = 42
torch.manual_seed(SEED)

# 3.读取数据

df = pd.read_csv("train.csv")

# 4.特征工程

# 从 Cabin 中提取 Deck
df["Deck"] = df["Cabin"].str[0]
df["Deck"] = df["Deck"].fillna("Unknown")

# 家庭人数
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 是否独自一人
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# 从 Name 中提取称呼
df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")

# 每个人对应的票价
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

# 5.划分训练集和测试集

# 取出标签
y = df["Survived"]

# 删除标签，得到特征
X = df.drop(columns=["Survived"])

# 随机打乱数据下标
indices = torch.randperm(len(df))

# 80%作为训练集
train_size = int(len(df) * 0.8)

# 从 PyTorch Tensor 转成 NumPy 数组
train_indices = indices[:train_size].numpy()
test_indices = indices[train_size:].numpy()

# 根据下标划分，并使用 copy() 避免 SettingWithCopyWarning
X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()

# 6.处理缺失值

# 年龄：使用训练集的中位数
age_median = X_train["Age"].median()

X_train["Age"] = X_train["Age"].fillna(age_median)
X_test["Age"] = X_test["Age"].fillna(age_median)

# Embarked：使用训练集的众数
embarked_mode = X_train["Embarked"].mode()[0]

X_train["Embarked"] = X_train["Embarked"].fillna(embarked_mode)
X_test["Embarked"] = X_test["Embarked"].fillna(embarked_mode)

# 7.删除暂时不使用的字段

drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]

X_train = X_train.drop(columns=drop_columns)
X_test = X_test.drop(columns=drop_columns)

# 8.独热编码

# 确定分类变量
categorical_features = ["Sex", "Embarked", "Deck", "Title"]

# 保存每个分类变量的类别
category_maps = {}

for col in categorical_features:

    # 只根据训练集确定类别
    categories = X_train[col].dropna().unique().tolist()

    category_maps[col] = categories

    # 测试集中没有见过的类别设为空
    X_test.loc[
        ~X_test[col].isin(categories),
        col
    ] = None

    X_train[col] = pd.Categorical(
        X_train[col],
        categories=categories
    )

    X_test[col] = pd.Categorical(
        X_test[col],
        categories=categories
    )

# 独热编码
X_train = pd.get_dummies(
    X_train,
    columns=categorical_features,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_features,
    dtype=int
)

# 保证测试集和训练集拥有完全相同的特征
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# 9.标准化数值特征

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "FarePerPerson"
]

# 只使用训练集计算平均值和标准差
train_mean = X_train[numeric_features].mean()
train_std = X_train[numeric_features].std()

# 防止某一列标准差为0
train_std = train_std.replace(0, 1)

# 训练集标准化
X_train[numeric_features] = (
    X_train[numeric_features] - train_mean
) / train_std

# 测试集使用训练集得到的参数
X_test[numeric_features] = (
    X_test[numeric_features] - train_mean
) / train_std

# 10.转换成 PyTorch Tensor

X_train_tensor = torch.tensor(
    X_train.values,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
)

# 把标签调整成 (N, 1) 的形状
y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1, 1)

# 11.创建 Dataset 和 DataLoader

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# 12.定义神经网络

class TitanicClassifier(nn.Module):
    def __init__(self,input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

# 根据实际处理后的特征数量确定输入层大小
input_size = X_train.shape[1]

model = TitanicClassifier(input_size)

# 13.定义损失函数和优化器

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

preprocessing_info = {
    "age_median": age_median,
    "embarked_mode": embarked_mode,
    "train_mean": train_mean.to_dict(),
    "train_std": train_std.to_dict(),
    "feature_columns": X_train.columns.tolist(),
    "categorical_features": categorical_features,
    "category_maps": category_maps
}

# 14.开始训练

# 设置训练轮数
epochs = 100

# 记录训练过程中的损失和准确率
loss_history = []
accuracy_history = []
test_accuracy_history = []

best_test_accuracy = 0.0
best_epoch = 0

# 连续多少轮没有提高就停止
patience = 20

# 记录已经连续多少轮没有提高
no_improve_count = 0

for epoch in range(epochs):

    # 训练
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        # 前向传播
        outputs = model(X_batch)

        # 计算损失
        loss = criterion(outputs, y_batch)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 记录损失
        total_loss += loss.item()

        # 计算准确率
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).float()

        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total

    loss_history.append(train_loss)
    accuracy_history.append(train_accuracy)

    # 测试
    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            outputs = model(X_batch)

            probabilities = torch.sigmoid(outputs)

            predictions = (probabilities >= 0.5).float()

            test_correct += (predictions == y_batch).sum().item()
            test_total += y_batch.size(0)

    test_accuracy = test_correct / test_total
    test_accuracy_history.append(test_accuracy)

    # 保存测试集表现最好的模型
    if test_accuracy > best_test_accuracy:

        # 更新最高准确率及最佳轮数
        best_epoch = epoch + 1
        best_test_accuracy = test_accuracy

        # 因为有提高，所以重新计数
        no_improve_count = 0

        print(
            f"发现新的最佳模型！"
            f"第 {epoch + 1} 轮，"
            f"测试准确率：{test_accuracy:.2%}"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": X_train.shape[1],
                "preprocessing": preprocessing_info,
                "best_test_accuracy": best_test_accuracy
            },
            "titanic_model_best.pth"
        )

    else:
        # 测试准确率没有提高
        no_improve_count += 1

    if no_improve_count >= patience:

        print()
        print(f"已经连续 {patience} 轮测试准确率没有提高，提前停止训练。")
        print(f"当前训练轮数：{epoch + 1}")
        print(f"目前最高测试准确率：{best_test_accuracy:.2%}")

        break

print()
print("训练完成！")
print(f"测试集最高准确率: {best_test_accuracy:.2%}")
print(f"最佳模型出现在第 {best_epoch} 轮")
print("最好的模型已经保存为:titanic_model_best.pth")

# 15.绘制训练过程曲线

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"

DOCS_DIR.mkdir(exist_ok=True)

# 图1：模型损失趋势图

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(loss_history) + 1),
    loss_history,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Trend")
plt.legend()
plt.grid(True)

# 自动调整布局
plt.tight_layout()

plt.savefig(DOCS_DIR / "loss_curve.png", dpi=300)

plt.show()

# 图2：训练集和测试集准确率

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(accuracy_history) + 1),
    accuracy_history,
    label="Training Accuracy"
)

plt.plot(
    range(1, len(test_accuracy_history) + 1),
    test_accuracy_history,
    label="Test Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(DOCS_DIR / "accuracy_curve.png", dpi=300)

plt.show()

# 16.重新加载模型

checkpoint = torch.load(
    "titanic_model_best.pth",
    weights_only=False
)

loaded_model = TitanicClassifier(
    checkpoint["input_size"]
)

loaded_model.load_state_dict(
    checkpoint["model_state_dict"]
)

loaded_model.eval()

print("模型重新加载成功！")