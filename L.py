import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 加载数据
train_data = pd.read_csv('D:\\train_data.csv')
test_data = pd.read_csv('D:\\test_data.csv')
dev_data = pd.read_csv('D:\\dev_data.csv')

# 检查数据
print(train_data.head())
print(dev_data.head())
print(test_data.head())

# 分离特征和目标变量
X_train = train_data[['cycle', 'capacity', 'resistance', 'CCCT', 'CVCT']]
y_train = train_data['SoH']

X_dev = dev_data[['cycle', 'capacity', 'resistance', 'CCCT', 'CVCT']]
y_dev = dev_data['SoH']

X_test = test_data[['cycle', 'capacity', 'resistance', 'CCCT', 'CVCT']]

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 在验证集上评估模型
y_pred_dev = model.predict(X_dev)
rmse = np.sqrt(mean_squared_error(y_dev, y_pred_dev))
print(f'Validation RMSE: {rmse}')

# 生成预测结果
y_pred_test = model.predict(X_test)

# 创建提交文件
submission = pd.DataFrame({
    'cycle': test_data['cycle'],
    'CS_Name': test_data['CS_Name'],
    'result': y_pred_test
})

# 保存为CSV文件
submission.to_csv('submission.csv', index=False)
