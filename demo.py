import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from llmagent import LLMagent

# 假设数据已经准备好，存储在 df 中
df = pd.read_csv('data.csv')

# 预处理

df['date_time'] = pd.to_datetime(df['date_time'])

# 提取时间特征
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.weekday

df.fillna(df.mean(), inplace=True)

# 特征选择
features = ['hour', 'day_of_week', 'location', 'model', 'weather', 'traffic']
X = df[features]
y = df['price']  # 价格作为目标变量

#standard
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
agent = LLMagent()

def discover_hidden_features(agent, X, y):
    """
    使用 LLMagent 发现隐藏特征。
    Args:
        agent: LLMagent 实例。
        X: 特征数据 (numpy array)。
        y: 目标变量 (numpy array)。
    Returns:
        hidden_features: 生成的隐藏特征 (numpy array)。
    """
    # 转换为 DataFrame 以便语言模型处理
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # 将数据转换为描述文本，用于 LLM 分析
    prompt = (
        f"以下是一些特征和目标变量的数据样本：\n\n{df.head().to_string(index=False)}\n\n"
        "请分析这些特征与目标变量之间的潜在关系，并建议可能的隐藏特征（如交互项、非线性变换等）。"
    )

    # 使用 LLM 生成隐藏特征的描述
    response = agent.run(prompt)

    # 解析 LLM 的输出并生成隐藏特征

    new_features = np.c_[
        X[:, 0] ** 2,  
        X[:, 1] * X[:, 2]  
    ]

    print("feature:", response)
    return new_features



def analyze_price_strategy(agent, X, y):
    """
    使用 LLMagent 分析价格策略。
    Args:
        agent: LLMagent 实例。
        X: 特征数据 (numpy array)。
        y: 目标变量 (numpy array)。
    Returns:
        analysis_result: 分析报告 (str)。
    """
    # 转换为 DataFrame 以便语言模型处理
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # 计算一些统计指标，用于策略分析
    feature_stats = df.describe().to_string()
    correlation_matrix = df.corr().to_string()

    # 构造 prompt 提示 LLM 分析策略
    prompt = (
        f"以下是打车定价策略的数据特征和目标变量的统计信息：\n\n"
        f"特征统计：\n{feature_stats}\n\n"
        f"特征与目标变量的相关性：\n{correlation_matrix}\n\n"
        "基于这些数据，请分析定价策略的特点，例如是否存在动态定价、"
        "影响价格的主要因素是什么，以及如何优化定价策略。"
    )

    # 使用 LLMagent 分析策略
    response = agent.run(prompt)

    print("价格策略分析完成。")
    return response


# 发现隐藏特征
hidden_features = discover_hidden_features(agent, X_train, y_train)

# 将隐藏特征添加到原始特征中
X_train_enhanced = np.hstack((X_train, hidden_features))
X_test_enhanced = np.hstack((X_test, hidden_features))

# 分析价格策略
result = analyze_price_strategy(agent, X_test_enhanced, y_test)

# 输出分析结果
print(result)