import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 1. 基本设置

SEED = 42
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "src" / "train.csv"

MODEL_PATH = BASE_DIR / "src" / "titanic_model_best.pth"

DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# 2. 读取数据

df = pd.read_csv(DATA_PATH)

print("数据读取成功！")
print(f"数据集大小：{df.shape}")

# 3. 特征工程

# 从 Cabin 中提取船舱甲板信息
df["Deck"] = df["Cabin"].str[0]
df["Deck"] = df["Deck"].fillna("Unknown")

# 家庭人数
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 是否独自一人
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# 从姓名中提取称呼
df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")

# 每个人承担的船票费用
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

# 4. 分离标签和特征

y = df["Survived"]
X = df.drop(columns=["Survived"])

# 5. 划分训练集、验证集、测试集

indices = torch.randperm(len(df))

train_size = int(len(df) * 0.64)
val_size = int(len(df) * 0.16)

train_indices = indices[:train_size].numpy()
val_indices = indices[train_size:train_size + val_size].numpy()
test_indices = indices[train_size + val_size:].numpy()

X_train = X.iloc[train_indices].copy()
X_val = X.iloc[val_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_val = y.iloc[val_indices].copy()
y_test = y.iloc[test_indices].copy()

print()
print("数据集划分完成：")
print(f"训练集：{len(X_train)} 条")
print(f"验证集：{len(X_val)} 条")
print(f"测试集：{len(X_test)} 条")

# 6. 缺失值处理

age_median = X_train["Age"].median()

X_train["Age"] = X_train["Age"].fillna(age_median)
X_val["Age"] = X_val["Age"].fillna(age_median)
X_test["Age"] = X_test["Age"].fillna(age_median)

embarked_mode = X_train["Embarked"].mode()[0]

X_train["Embarked"] = X_train["Embarked"].fillna(embarked_mode)
X_val["Embarked"] = X_val["Embarked"].fillna(embarked_mode)
X_test["Embarked"] = X_test["Embarked"].fillna(embarked_mode)

# 7. 删除不直接使用的原始字段

drop_columns = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

X_train = X_train.drop(columns=drop_columns)
X_val = X_val.drop(columns=drop_columns)
X_test = X_test.drop(columns=drop_columns)

# 8. 类别特征处理

categorical_features = [
    "Sex",
    "Embarked",
    "Deck",
    "Title"
]

category_maps = {}

for col in categorical_features:

    # 只能使用训练集确定有哪些类别
    categories = X_train[col].dropna().unique().tolist()

    category_maps[col] = categories

    # 验证集出现训练集中没有的类别时，设为空
    X_val.loc[
        ~X_val[col].isin(categories),
        col
    ] = None

    # 测试集出现训练集中没有的类别时，设为空
    X_test.loc[
        ~X_test[col].isin(categories),
        col
    ] = None

    # 使用训练集确定的类别范围
    X_train[col] = pd.Categorical(
        X_train[col],
        categories=categories
    )

    X_val[col] = pd.Categorical(
        X_val[col],
        categories=categories
    )

    X_test[col] = pd.Categorical(
        X_test[col],
        categories=categories
    )

# 9. One-Hot 独热编码

X_train = pd.get_dummies(
    X_train,
    columns=categorical_features,
    dtype=int
)

X_val = pd.get_dummies(
    X_val,
    columns=categorical_features,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_features,
    dtype=int
)

# 保证验证集和测试集的特征列
# 与训练集完全一致
X_val = X_val.reindex(
    columns=X_train.columns,
    fill_value=0
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# 10. 数值特征标准化

# 均值和标准差只使用训练集计算。
# 验证集和测试集使用训练集的参数。

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

train_mean = X_train[numeric_features].mean()
train_std = X_train[numeric_features].std()

# 如果某一列标准差为 0，避免除以 0
train_std = train_std.replace(0, 1)

X_train[numeric_features] = (
    X_train[numeric_features] - train_mean
) / train_std

X_val[numeric_features] = (
    X_val[numeric_features] - train_mean
) / train_std

X_test[numeric_features] = (
    X_test[numeric_features] - train_mean
) / train_std

# 11. 转换成 PyTorch Tensor

X_train_tensor = torch.tensor(
    X_train.values,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val.values,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test.values,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)

y_val_tensor = torch.tensor(
    y_val.values,
    dtype=torch.float32
).reshape(-1, 1)

y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1, 1)

print()
print(f"模型输入特征数量：{X_train.shape[1]}")

# 12. 创建 Dataset 和 DataLoader

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
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

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# 13. 定义神经网络

class TitanicClassifier(nn.Module):

    def __init__(self, input_size):

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


input_size = X_train.shape[1]

model = TitanicClassifier(input_size)

# 14. 定义损失函数和优化器

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 15. 训练参数

epochs = 100

patience = 20

best_val_accuracy = 0.0

no_improve_count = 0

loss_history = []

train_accuracy_history = []

val_accuracy_history = []

# 16. 定义评估函数

# 这里专门负责计算准确率。
def evaluate_accuracy(model, data_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            outputs = model(X_batch)

            probabilities = torch.sigmoid(outputs)

            predictions = (
                probabilities >= 0.5
            ).float()

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

    return correct / total

# 17. 开始训练

for epoch in range(epochs):

    # 训练阶段

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        # 前向传播
        outputs = model(X_batch)

        # 计算损失
        loss = criterion(
            outputs,
            y_batch
        )

        # 清空之前的梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()


    # 计算本轮平均训练损失
    train_loss = (
        total_loss / len(train_loader)
    )

    loss_history.append(train_loss)

    # 训练集准确率
    
    train_accuracy = evaluate_accuracy(
        model,
        train_loader
    )

    train_accuracy_history.append(
        train_accuracy
    )

    # 验证集准确率

    val_accuracy = evaluate_accuracy(
        model,
        val_loader
    )

    val_accuracy_history.append(
        val_accuracy
    )


    print(
        f"第 {epoch + 1:03d} 轮 | "
        f"Loss: {train_loss:.4f} | "
        f"训练准确率: {train_accuracy:.2%} | "
        f"验证准确率: {val_accuracy:.2%}"
    )

    # 根据验证集选择最佳模型

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        no_improve_count = 0

        print(
            f"发现新的最佳模型！"
            f"验证准确率：{val_accuracy:.2%}"
        )

        # 保存预处理信息
        preprocessing_info = {

            "age_median": age_median,

            "embarked_mode": embarked_mode,

            "train_mean": train_mean.to_dict(),

            "train_std": train_std.to_dict(),

            "feature_columns": X_train.columns.tolist(),

            "categorical_features": categorical_features,

            "category_maps": category_maps
        }

        # 保存模型
        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "input_size":
                    X_train.shape[1],

                "preprocessing":
                    preprocessing_info,

                "best_val_accuracy":
                    best_val_accuracy
            },
            MODEL_PATH
        )


    else:

        no_improve_count += 1

    # 早停

    if no_improve_count >= patience:

        print()

        print(
            f"已经连续 {patience} 轮"
            f"验证集准确率没有提高，"
            f"提前停止训练。"
        )

        print(
            f"当前训练轮数：{epoch + 1}"
        )

        print(
            f"目前最高验证集准确率："
            f"{best_val_accuracy:.2%}"
        )

        break

# 18. 训练结束

print()

print("训练完成！")

print(
    f"验证集最高准确率："
    f"{best_val_accuracy:.2%}"
)

print(
    f"最佳模型已经保存到："
    f"{MODEL_PATH}"
)

# 19. 重新加载验证集上表现最好的模型

checkpoint = torch.load(
    MODEL_PATH,
    weights_only=False
)

best_model = TitanicClassifier(
    checkpoint["input_size"]
)

best_model.load_state_dict(
    checkpoint["model_state_dict"]
)

best_model.eval()


print()
print("最佳模型重新加载成功！")

# 20. 最终测试

final_test_accuracy = evaluate_accuracy(
    best_model,
    test_loader
)

print()
print(
    f"最终测试集准确率："
    f"{final_test_accuracy:.2%}"
)

# 21. 绘制 Loss 曲线

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

plt.tight_layout()

plt.savefig(
    DOCS_DIR / "loss_curve.png",
    dpi=300
)

plt.show()

# 22. 绘制训练集和验证集准确率曲线

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(train_accuracy_history) + 1),
    train_accuracy_history,
    label="Training Accuracy"
)

plt.plot(
    range(1, len(val_accuracy_history) + 1),
    val_accuracy_history,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    DOCS_DIR / "accuracy_curve.png",
    dpi=300
)

plt.show()

# 23. 最终信息

print()
print("项目训练完成")
print(
    f"最佳验证集准确率："
    f"{best_val_accuracy:.2%}"
)

print(
    f"最终测试集准确率："
    f"{final_test_accuracy:.2%}"
)

print(
    f"模型文件：{MODEL_PATH}"
)

print(
    f"Loss 曲线："
    f"{DOCS_DIR / 'loss_curve.png'}"
)

print(
    f"准确率曲线："
    f"{DOCS_DIR / 'accuracy_curve.png'}"
)