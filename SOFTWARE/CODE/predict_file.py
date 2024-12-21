import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler  # 使用 MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ----------------------------
# 定义物理方程和常量
# ----------------------------
R = 8.314462618  # 理想气体常数，单位 J/(mol·K)


def calculate_a_prime(Theta, Tc, Pc):
    return 0.42748 * (Theta ** 2) * (R ** 2) * (Tc ** 2) / Pc


def calculate_b_prime(Theta, Tc, Pc):
    return 0.08664 * Theta * R * Tc / Pc


def calculate_pressure(Theta, T, V, Pc, Tc):
    a_prime = calculate_a_prime(Theta, Tc, Pc)
    b_prime = calculate_b_prime(Theta, Tc, Pc)
    P = (Theta * R * T) / (V - b_prime) - a_prime / (V * (V + b_prime))
    return P


# ----------------------------
# 加载保存的模型和预处理对象
# ----------------------------
model = joblib.load('best_mlp_model_theta.joblib')
scaler = joblib.load('scaler_theta.joblib')
imputer = joblib.load('imputer_theta.joblib')

# ----------------------------
# 定义每个特征的最小值、最大值和平均值
# ----------------------------
feature_stats = {
    'Tr': {'mean': 0.567318216, 'max': 0.872353937, 'min': 0.30875535},
    'Pr': {'mean': 0.473201096, 'max': 0.996677741, 'min': 0.047192072},
    'LIQUID PL WATER': {'mean': 36098.51718, 'max': 751750, 'min': 0.00127681},
    'LIQUID K WATER': {'mean': 0.150305589, 'max': 0.255235, 'min': 0.0844824},
    'LIQUID MU WATER': {'mean': 0.0064721, 'max': 0.1978055, 'min': 0.000120686},
    'LIQUID SIGMA WATER': {'mean': 0.023416673, 'max': 0.0389797, 'min': 0.00656363},
    'TOTAL RHO WATER': {'mean': 879.7665995, 'max': 1098.303, 'min': 656.8451},
    'LIQUID DHVL WATER': {'mean': 51287.62828, 'max': 89053.24, 'min': 23194.53},
    'M': {'mean': 107.4025125, 'max': 172.308, 'min': 58.0791},
    'Tc': {'mean': 621.95125, 'max': 708.6, 'min': 545.1},
    'Pc': {'mean': 3932375, 'max': 5786000, 'min': 2119000},
    'FREEZEPT': {'mean': 245.6575, 'max': 333.15, 'min': 144.15},
    'TB': {'mean': 441.49125, 'max': 562.15, 'min': 370.23},
    'omiga': {'mean': 0.534346375, 'max': 0.706632, 'min': 0.369047},
}

# ----------------------------
# 准备固定特征的固定值
# ----------------------------
features = [
    'Tr', 'Pr', 'LIQUID PL', 'LIQUID K', 'LIQUID MU', 'LIQUID SIGMA',
    'TOTAL RHO', 'LIQUID DHVL', 'M', 'Tc', 'Pc',
    'FREEZEPT', 'TB', 'omiga'
]

fixed_features = {feat: 0.5 for feat in features}  # 其他特征固定为0.5

def sensitivity_analysis(feature_name, model, imputer, scaler, features, fixed_features, num_points=10000,
                         poly_degree=3):
    """
    对单个特征进行敏感性分析。

    Parameters:
    - feature_name: str, 特征名称
    - model: 已训练的模型
    - imputer: SimpleImputer 对象
    - scaler: MinMaxScaler 对象
    - features: list, 特征列表
    - fixed_features: dict, 其他特征固定值（设置为0.5）
    - num_points: int, 生成的数据点数
    - poly_degree: int, 多项式拟合的阶数

    Returns:
    - None (保存图像和拟合方程)
    """
    # 创建一个DataFrame，其中当前特征从0到1变化，其他特征固定为0.5
    data_dict = {}
    for feat in features:
        if feat == feature_name:
            data_dict[feat] = np.linspace(0, 1, num_points)  # 归一化后0-1
        else:
            data_dict[feat] = np.full(num_points, fixed_features[feat])  # 固定为0.5

    df_varied = pd.DataFrame(data_dict)

    # 预处理数据
    X_varied = df_varied[features]
    print("当前特征列:", X_varied.columns)
    print("模型训练时的特征列:", features)
    X_varied_imputed = pd.DataFrame(imputer.transform(X_varied), columns=features)

    # 预测 Theta
    Theta_pred = model.predict(X_varied_imputed)
    df_varied['Theta_pred'] = Theta_pred

    # 保存生成的数据
    output_csv = f'sensitivity_{feature_name}.csv'
    df_varied.to_csv(output_csv, index=False)
    print(f"生成的敏感性分析数据已保存到 {output_csv}")

    # 绘制 Theta 随该特征变化的关系图
    plt.figure(figsize=(10, 6))
    plt.scatter(df_varied[feature_name], Theta_pred, s=1, alpha=0.5, label='Predicted Theta')
    plt.xlabel(feature_name)
    plt.ylabel('Theta_pred')
    plt.title(f'Theta vs {feature_name}')

    # 多项式拟合
    poly = PolynomialFeatures(degree=poly_degree)
    X_poly = poly.fit_transform(df_varied[[feature_name]])
    poly_model = LinearRegression()
    poly_model.fit(X_poly, Theta_pred)
    Theta_fit = poly_model.predict(X_poly)

    # 绘制拟合曲线
    plt.plot(df_varied[feature_name], Theta_fit, color='red', label=f'Polynomial fitting (degree={poly_degree})')
    plt.legend()
    plt.tight_layout()

    # 保存图像
    plot_filename = f'sensitivity_{feature_name}.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Theta 随 {feature_name} 变化的关系图已保存为 {plot_filename}")

    # 输出拟合方程
    coeffs = poly_model.coef_
    intercept = poly_model.intercept_
    equation = f'Theta = {intercept:.4f}'
    for i in range(1, len(coeffs)):
        equation += f' + {coeffs[i]:.4f}*{feature_name}^{i}'
    print(f'{feature_name} 的多项式拟合方程（degree={poly_degree}）:')
    print(equation)

    # 保存拟合方程到文本文件
    with open(f'sensitivity_{feature_name}_fit_equation.txt', 'w') as f:
        f.write(f'多项式拟合方程 (degree={poly_degree}):\n')
        f.write(equation)
    print(f'拟合方程已保存到 sensitivity_{feature_name}_fit_equation.txt\n')


# ----------------------------
# 执行敏感性分析
# ----------------------------
for feature in features:
    print(f"开始对特征 {feature} 进行敏感性分析...")
    sensitivity_analysis(
        feature_name=feature,
        model=model,
        imputer=imputer,
        scaler=scaler,
        features=features,
        fixed_features=fixed_features,
        num_points=10000,
        poly_degree=5  # 可以根据需要调整多项式阶数
    )
