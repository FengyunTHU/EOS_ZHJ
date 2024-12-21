# 这是机器学习训练脚本

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error
from bayes_opt import BayesianOptimization
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # 用于保存模型
from tqdm import tqdm  # 进度条

# 忽略警告信息（可选）
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 步骤 1：读取并合并Excel文件的数据
# ----------------------------
data_frames = []

# 读取 'theta_data_water_1.xlsx' 的第一个 sheet
df = pd.read_excel('theta_data_water_1.xlsx', sheet_name=0)
data_frames.append(df)

# 读取 '酮theta.xlsx' 的前五个 sheets
for sheet_name in range(4):
    df = pd.read_excel('酮theta.xlsx', sheet_name=sheet_name)
    data_frames.append(df)

for sheet_name in range(7):
    df = pd.read_excel('醇酸theta.xlsx', sheet_name=sheet_name)
    data_frames.append(df)

# 合并所有数据框
data = pd.concat(data_frames, ignore_index=True)

# ----------------------------
# 步骤 2：处理缺失值
# ----------------------------
# 将所有空字符串替换为 NaN
data.replace('', np.nan, inplace=True)

# 定义特征和目标变量
features = [
    'Tr', 'Pr', 'LIQUID PL', 'LIQUID K', 'LIQUID MU', 'LIQUID SIGMA',
    'TOTAL RHO', 'LIQUID DHVL', 'M', 'Tc', 'Pc',
    'FREEZEPT', 'TB', 'omiga'
]
target = 'theta'  # 确保 'theta' 是目标变量的准确列名

# 检查特征和目标变量是否存在
missing_features = [feature for feature in features if feature not in data.columns]
if missing_features:
    raise ValueError(f"缺失必要的特征列: {missing_features}")
if target not in data.columns:
    raise ValueError(f"缺失目标变量列: {target}")

# 使用 SimpleImputer 进行均值填充，仅针对特征列
imputer = SimpleImputer(strategy='mean')
imputer.fit(data[features])  # 仅拟合特征列

# 仅转换特征列
X_imputed = pd.DataFrame(imputer.transform(data[features]), columns=features)
y = data[target]

# ----------------------------
# 步骤 3：数据标准化
# ----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ----------------------------
# 步骤 4：多折交叉验证
# ----------------------------
# 定义5折交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 定义评估指标
scorer_mse = make_scorer(mean_squared_error, greater_is_better=False)
scorer_r2 = make_scorer(r2_score)

# ----------------------------
# 步骤 5：贝叶斯优化进行超参数优化
# ----------------------------
# 定义超参数的搜索范围和目标函数
def mlp_cv(hidden_layer_sizes, alpha, learning_rate_init):
    hidden_layer_sizes = int(hidden_layer_sizes)
    alpha = 10 ** -alpha
    learning_rate_init = 10 ** -learning_rate_init

    model = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_sizes,),
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=2000,
        random_state=42
    )

    # 使用负均方误差作为优化目标（贝叶斯优化默认是寻找最大值）
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scorer_mse, n_jobs=-1)
    return scores.mean()

# 设置参数的范围
pbounds = {
    'hidden_layer_sizes': (50, 200),
    'alpha': (3, 7),  # 10^-3 to 10^-7
    'learning_rate_init': (3, 7)  # 10^-3 to 10^-7
}

# 进行贝叶斯优化
optimizer = BayesianOptimization(
    f=mlp_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

print("开始贝叶斯优化...")
optimizer.maximize(init_points=10, n_iter=15)

# 获取最佳参数
best_params = optimizer.max['params']
best_hidden_layer_sizes = int(best_params['hidden_layer_sizes'])
best_alpha = 10 ** -best_params['alpha']
best_learning_rate_init = 10 ** -best_params['learning_rate_init']

print("\n最佳参数：")
print(f"hidden_layer_sizes: {best_hidden_layer_sizes}")
print(f"alpha: {best_alpha}")
print(f"learning_rate_init: {best_learning_rate_init}")

# ----------------------------
# 步骤 6：训练神经网络模型，预测 Theta
# ----------------------------
# 使用最佳参数训练最终模型
best_model = MLPRegressor(
    hidden_layer_sizes=(best_hidden_layer_sizes,),
    alpha=best_alpha,
    learning_rate_init=best_learning_rate_init,
    max_iter=2000,
    random_state=42
)

print("\n训练最佳模型...")
best_model.fit(X_scaled, y)

# 预测 Theta
Theta_pred = best_model.predict(X_scaled)

# ----------------------------
# 步骤 7：评估模型性能
# ----------------------------
# 计算预测 Theta 与实际 Theta 之间的均方误差和其他指标
mse_pressure = mean_squared_error(y, Theta_pred)
mae_pressure = mean_absolute_error(y, Theta_pred)
r2_pressure = r2_score(y, Theta_pred)

print("\n压力预测性能评估:")
print(f"均方误差(MSE): {mse_pressure:.4f}")
print(f"平均绝对误差(MAE): {mae_pressure:.4f}")
print(f"决定系数(R²): {r2_pressure:.4f}")

# ----------------------------
# 步骤 9：保存训练好的模型和预处理对象
# ----------------------------
print("\n保存模型和预处理对象...")
# 保存模型
joblib.dump(best_model, 'best_mlp_model_theta.joblib')

# 保存标准化器
joblib.dump(scaler, 'scaler_theta.joblib')

# 保存Imputer
joblib.dump(imputer, 'imputer_theta.joblib')

# ----------------------------
# 步骤 11：使用SHAP分析评估特征重要性
# ----------------------------
# 计算SHAP值
print("\n计算SHAP值...")
# **修正：在 SHAP 分析中使用原始的 X 数据框，并传递特征名称**
explainer = shap.Explainer(best_model.predict, X_scaled, feature_names=features)
shap_values = explainer(X_scaled)

# 保存SHAP柱状图（特征重要性）
print("保存SHAP柱状图...")
shap.summary_plot(shap_values, features=features, plot_type="bar", show=False)
plt.title('SHAP Summary (Bar)')
plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图像，防止后续显示问题

# 保存SHAP点图（详细分布）
print("保存SHAP点图...")
shap.summary_plot(shap_values, X_scaled, show=False, )
plt.title('SHAP Summary (Dot)')
plt.savefig('shap_summary_dot.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------
# 步骤 10：多折交叉验证评估模型稳定性
# ----------------------------
print("\n进行交叉验证评估...")
cv_scores_mse = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
cv_scores_mae = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
cv_scores_r2 = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='r2')

cv_mse = -cv_scores_mse.mean()
cv_std_mse = cv_scores_mse.std()

cv_mae = -cv_scores_mae.mean()
cv_std_mae = cv_scores_mae.std()

cv_r2 = cv_scores_r2.mean()
cv_std_r2 = cv_scores_r2.std()

print(f"交叉验证均方误差（MSE）：{cv_mse:.4f} ± {cv_std_mse:.4f}")
print(f"交叉验证平均绝对误差（MAE）：{cv_mae:.4f} ± {cv_std_mae:.4f}")
print(f"交叉验证决定系数（R²）：{cv_r2:.4f} ± {cv_std_r2:.4f}")

# ----------------------------
# 步骤 11：绘制交叉验证的结果
# ----------------------------
# 绘制交叉验证的MSE分布
plt.figure(figsize=(10, 6))
sns.boxplot(data=-cv_scores_mse)
plt.title('Cross-Validation MSE Distribution')
plt.ylabel('MSE')
plt.savefig('cv_mse_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制交叉验证的MAE分布
plt.figure(figsize=(10, 6))
sns.boxplot(data=-cv_scores_mae)
plt.title('Cross-Validation MAE Distribution')
plt.ylabel('MAE')
plt.savefig('cv_mae_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制交叉验证的R²分布
plt.figure(figsize=(10, 6))
sns.boxplot(data=cv_scores_r2)
plt.title('Cross-Validation R² Distribution')
plt.ylabel('R²')
plt.savefig('cv_r2_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------
# 步骤 12：绘制预测结果对比图
# ----------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y, y=Theta_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('实际压力 vs. 预测压力')
plt.xlabel('实际压力 (Pa)')
plt.ylabel('预测压力 (Pa)')
plt.savefig('pressure_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n所有步骤完成，模型和图像已保存。")

