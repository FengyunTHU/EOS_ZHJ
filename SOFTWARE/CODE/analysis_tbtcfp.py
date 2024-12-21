import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 请根据实际的特征列表与环境修改这些名称
features = [
    'Tr', 'Pr', 'LIQUID PL', 'LIQUID K', 'LIQUID MU', 'LIQUID SIGMA',
    'TOTAL RHO', 'LIQUID DHVL', 'M', 'Tc', 'Pc',
    'FREEZEPT', 'TB', 'omiga'
]

model = joblib.load('best_mlp_model_theta.joblib')
imputer = joblib.load('imputer_theta.joblib')
scaler = joblib.load('scaler_theta.joblib')

# 假设其他特征固定为0.5
fixed_features = {feat: 0.5 for feat in features}

def predict_theta(data_df):
    # 对传入数据进行缺失值填充和缩放
    X_imputed = pd.DataFrame(imputer.transform(data_df), columns=features)
    X_scaled = scaler.transform(X_imputed)
    Theta_pred = model.predict(X_scaled)
    return Theta_pred

def two_feature_sensitivity(feature_x, feature_y, num_points=50):
    # 两个特征从0到1变化，形成网格
    x_values = np.linspace(0, 1, num_points)
    y_values = np.linspace(0, 1, num_points)
    X_grid, Y_grid = np.meshgrid(x_values, y_values)

    data_dict = {}
    for feat in features:
        if feat == feature_x:
            data_dict[feat] = X_grid.ravel()
        elif feat == feature_y:
            data_dict[feat] = Y_grid.ravel()
        else:
            data_dict[feat] = np.full(num_points * num_points, 0.5)

    df_varied = pd.DataFrame(data_dict)
    Theta_pred = predict_theta(df_varied)
    Z = Theta_pred.reshape(X_grid.shape)

    # 将数据保存
    df_varied['Theta_pred'] = Theta_pred
    df_varied.to_csv(f'two_sensitivity_{feature_x}_{feature_y}.csv', index=False)

    # 绘制等高线填充图
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X_grid, Y_grid, Z, levels=50, cmap='viridis')
    # 添加颜色条
    cbar = plt.colorbar(contour)
    cbar.set_label('Theta_pred', rotation=270, labelpad=15)

    # 在等高线图上叠加散点
    # 散点位置与网格点相同，这里用圆圈表示，并用edgecolor加边框以形成类似您示例中的圆点外框效果
    # s=40可以根据需要调整点大小
    plt.scatter(X_grid, Y_grid, c=Z, s=40, edgecolor='black', linewidth=0.5, cmap='viridis')

    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'Theta vs {feature_x} & {feature_y} (others=0.5)')
    plt.tight_layout()
    plt.savefig(f'two_sensitivity_{feature_x}_{feature_y}.png', dpi=300)
    plt.close()

# 执行两特征变化分析示例
two_feature_sensitivity('TB', 'Tc', num_points=50)
two_feature_sensitivity('TB', 'FREEZEPT', num_points=50)
two_feature_sensitivity('Tc', 'FREEZEPT', num_points=50)
