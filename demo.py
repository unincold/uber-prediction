import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from llmagent import LLMagent

def get_price(hour, day_of_week, location, model, weather, traffic):
    
    #use uber api to get the price
    #price = uber_api.get_price(hour, day_of_week, location, model, weather, traffic)
    return price


####df is not prepared
df = pd.read_csv('data.csv')

# 预处理

df['date_time'] = pd.to_datetime(df['date_time'])

df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.weekday

df.fillna(df.mean(), inplace=True)

# features and target
features = ['hour', 'day_of_week', 'location', 'model', 'weather', 'traffic']
X = df[features]
y = df['price']  

#standard
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
agent = LLMagent()
def response_to_features(response, X):
    """
    将 LLM 响应解析为隐藏特征。
    Args:
        response: LLM 的文本响应 (str)。
        X: 原始特征数据 (numpy array)。
    Returns:
        new_features: 生成的隐藏特征 (numpy array)。
    """
    hidden_features = []
    for match in re.finditer(r"特征\s*(\d+)\s*(的平方|和特征\s*(\d+)\s*的乘积|的对数)", response):
        feature_idx = int(match.group(1))
        operation = match.group(2)
        
        if operation == "的平方":
            hidden_features.append(X[:, feature_idx] ** 2)
        elif "乘积" in operation and match.group(3):
            second_feature_idx = int(match.group(3))
            hidden_features.append(X[:, feature_idx] * X[:, second_feature_idx])
        elif operation == "的对数":
            safe_values = np.maximum(X[:, feature_idx], 1e-6)
            hidden_features.append(np.log(safe_values))
    
    if hidden_features:
        return np.stack(hidden_features, axis=1)
    else:
        return np.empty((X.shape[0], 0))

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
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # can change the prompt to English
    prompt = (
        f"these are the features and target variables:\n\n{df.head().to_string(index=False)}\n\n"
        "please discover the hidden features and return the result in standard format."
        
    )

    # 使用 LLM 生成隐藏特征的描述
    response = agent.run(prompt)

    #response->new_features is not finished
    #missing_part
    #below is demo


    new_features = response_to_features(response, X)

    print("feature:", response)
    return new_features

def analyze_weight_of_features(agent, X, y):
    #if a feature changes 1%, how much the price will change
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    #ask the agent to analyze the weight of the features by changing 1%
    prompt = (
        f"these are the features and target variables:\n\n{df.head().to_string(index=False)}\n\n"
        "please analyze the weight of the features by changing 1% and return the result in standard format."
    )
    response = agent.run(prompt)
    #deal with the response and output the format
    return response


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


    feature_stats = df.describe().to_string()
    correlation_matrix = df.corr().to_string()

    # 构造 prompt 提示 LLM 分析策略
    prompt = (
        f"these are the features and target variables:\n\n{df.head().to_string(index=False)}\n\n"
        f"feature statistics:\n\n{feature_stats}\n\n"
        f"correlation matrix:\n\n{correlation_matrix}\n\n"
        "please analyze the price strategy and return the result in standard format."
        
    )

    # 使用 LLMagent 分析策略
    response = agent.run(prompt)

    print("done")
    return response



hidden_features = discover_hidden_features(agent, X_train, y_train)
#1 relations
relation_Xtrain_and_hidden = np.corrcoef(X_train.T, hidden_features.T)
if (relation_Xtrain_and_hidden > 0.5).all():
    #redo the prompt
    hidden_features = discover_hidden_features(agent, X_train, y_train)
# 将隐藏特征添加到原始特征中
X_train_enhanced = np.hstack((X_train, hidden_features))
X_test_enhanced = np.hstack((X_test, hidden_features))
#analyse hidden features with other features not listed in the prompt
#print the probility the price is related to the hidden features and some data related to the hidden features
'''undo'''
'''goal: to find the hidden features that are related to the price'''
# 分析价格策略

result = analyze_price_strategy(agent, X_test_enhanced, y_test)

# 输出分析结果
print(result)