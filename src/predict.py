# 1.导入必要的库
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# 模型文件所在文件夹
BASE_DIR = Path(__file__).resolve().parent

# 模型文件路径
MODEL_PATH = BASE_DIR / "titanic_model_best.pth"

# 2.定义和训练时完全一样的模型
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

# 3.加载训练好的模型
try:
    checkpoint = torch.load(
        MODEL_PATH,
        weights_only=False,
        map_location="cpu"
    )

except Exception as e:
    print("无法加载模型文件。")
    print("请确认 titanic_model_best.pth 和 predict.py 在同一个文件夹中。")
    print(f"错误信息：{e}")
    input("按回车键退出...")
    raise SystemExit

preprocessing = checkpoint["preprocessing"]

loaded_model = TitanicClassifier(
    checkpoint["input_size"]
)

loaded_model.load_state_dict(
    checkpoint["model_state_dict"]
)

loaded_model.eval()

# 4.新乘客预测函数
def predict_passenger(
    Pclass,
    Sex,
    Age,
    SibSp,
    Parch,
    Fare,
    Embarked,
    Deck,
    Title
):

    # 计算特征工程
    FamilySize = SibSp + Parch + 1

    IsAlone = int(FamilySize == 1)

    FarePerPerson = Fare / FamilySize

    # 创建一个 DataFrame 来存储新乘客的数据
    passenger = pd.DataFrame([{
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Embarked": Embarked,
        "Deck": Deck,
        "FamilySize": FamilySize,
        "IsAlone": IsAlone,
        "Title": Title,
        "FarePerPerson": FarePerPerson
    }])

    # 缺失值处理
    passenger["Age"] = passenger["Age"].fillna(
        preprocessing["age_median"]
    )

    passenger["Embarked"] = passenger["Embarked"].fillna(
        preprocessing["embarked_mode"]
    )

    # 分类变量处理
    categorical_features = preprocessing[
        "categorical_features"
    ]

    # 获取类别消息
    category_maps = preprocessing[
        "category_maps"
    ]

    for col in categorical_features:

        categories = category_maps[col]

        if passenger.loc[0, col] not in categories:
            passenger.loc[0, col] = None

        passenger[col] = pd.Categorical(
            passenger[col],
            categories=categories
        )

    # 独热编码
    passenger = pd.get_dummies(
        passenger,
        columns=categorical_features,
        dtype=int
    )

    # 保证特征顺序和训练数据完全一致
    passenger = passenger.reindex(
        columns=preprocessing["feature_columns"],
        fill_value=0
    )

    # 标准化
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

    train_mean = pd.Series(
        preprocessing["train_mean"]
    )

    train_std = pd.Series(
        preprocessing["train_std"]
    )

    passenger[numeric_features] = (
        passenger[numeric_features] - train_mean
    ) / train_std

    # 转换成 Tensor
    passenger_tensor = torch.tensor(
        passenger.values,
        dtype=torch.float32
    )

    # 模型预测
    with torch.no_grad():

        output = loaded_model(
            passenger_tensor
        )

        probability = torch.sigmoid(
            output
        ).item()

        prediction = int(
            probability >= 0.5
        )

    return prediction, probability

# 5.以下部分负责终端互动输入
def input_pclass():
    while True:
        value = input(
            "舱位等级 Pclass(1/2/3):"
        ).strip()

        try:
            value = int(value)

            if value in [1, 2, 3]:
                return value

            print("输入错误，请输入 1、2 或 3。")

        except ValueError:
            print("输入错误，请输入数字 1、2 或 3。")

def input_sex():
    while True:
        value = input(
            "性别 Sex(male/female):"
        ).strip().lower()

        if value in ["male", "m"]:
            return "male"

        if value in ["female", "f"]:
            return "female"

        print("输入错误，请输入 male 或 female。")
        print("大小写不影响，例如 Male、MALE、male 都可以。")

def input_age():
    while True:
        value = input(
            "年龄 Age:"
        ).strip()

        try:
            age = float(value)

            if 0 <= age <= 200:
                return age

            print("输入错误，年龄应该在 0 到 200 之间。")

        except ValueError:
            print("输入错误，请输入数字，例如 25 或 25.5。")

def input_sibsp():
    while True:
        value = input(
            "兄弟姐妹/配偶数量 SibSp:"
        ).strip()

        try:
            value = int(value)

            if value >= 0:
                return value

            print("输入错误,SibSp 不能小于 0。")

        except ValueError:
            print("输入错误，请输入非负整数，例如 0、1、2。")

def input_parch():
    while True:
        value = input(
            "父母/子女数量 Parch:"
        ).strip()

        try:
            value = int(value)

            if value >= 0:
                return value

            print("输入错误,Parch 不能小于 0。")

        except ValueError:
            print("输入错误，请输入非负整数，例如 0、1、2。")

def input_fare():
    while True:
        value = input(
            "票价 Fare:"
        ).strip()

        try:
            fare = float(value)

            if fare >= 0:
                return fare

            print("输入错误，票价不能小于 0。")

        except ValueError:
            print("输入错误，请输入数字，例如 10 或 32.5。")

def input_embarked():
    while True:
        value = input(
            "登船港口 Embarked(C/Q/S):"
        ).strip().upper()

        if value in ["C", "Q", "S"]:
            return value

        print("输入错误，请输入 C、Q 或 S。")
        print("大小写不影响，例如 c、C 都可以。")

def input_deck():

    # 从训练时保存的类别中读取 Deck 选项
    deck_categories = preprocessing["category_maps"]["Deck"]

    # 显示给用户看的选项
    display_options = "/".join(deck_categories)

    # 建立大小写不敏感的映射
    deck_map = {
        str(category).lower(): category
        for category in deck_categories
    }

    # 额外允许用户输入 u 表示 Unknown
    if "unknown" in deck_map:
        deck_map["u"] = deck_map["unknown"]

    while True:

        value = input(
            f"甲板 Deck({display_options}):"
        ).strip().lower()

        if value in deck_map:
            return deck_map[value]

        print(f"输入错误，请输入：{display_options}")
        print("大小写不影响，例如 a、A 都可以。")

def input_title():

    # 从训练时保存的类别中读取 Title
    title_categories = preprocessing["category_maps"]["Title"]

    # 建立大小写不敏感的映射
    title_map = {
        str(title).lower(): title
        for title in title_categories
    }

    display_options = "/".join(title_categories)

    while True:

        value = input(
            f"称谓 Title({display_options}):"
        ).strip().lower()

        if value in title_map:
            return title_map[value]

        print("输入错误，请输入训练数据中存在的称谓。")
        print(f"可输入：{display_options}")
        print("大小写不影响，例如 mr、MR、Mr 都可以。")

# 6.终端互动预测
def interactive_predict():

    print()
    print("=" * 45)
    print("       泰坦尼克号乘客生存预测")
    print("=" * 45)
    print("请输入乘客信息。")
    print("字母输入不区分大小写。")
    print()

    Pclass = input_pclass()

    Sex = input_sex()

    Age = input_age()

    SibSp = input_sibsp()

    Parch = input_parch()

    Fare = input_fare()

    Embarked = input_embarked()

    Deck = input_deck()

    Title = input_title()

    # 模型预测
    prediction, probability = predict_passenger(
        Pclass=Pclass,
        Sex=Sex,
        Age=Age,
        SibSp=SibSp,
        Parch=Parch,
        Fare=Fare,
        Embarked=Embarked,
        Deck=Deck,
        Title=Title
    )

    print()
    print("=" * 45)
    print("                 预测结果")
    print("=" * 45)

    print(
        f"模型预测幸存概率：{probability:.2%}"
    )

    if prediction == 1:
        print("预测结果:幸存(1)")
    else:
        print("预测结果:未幸存(0)")

    print("=" * 45)

# 只有在直接运行 predict.py 时才会执行交互式预测
if __name__ == "__main__":
    interactive_predict()