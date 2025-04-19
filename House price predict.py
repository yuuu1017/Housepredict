import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 載入資料
df = pd.read_csv('Taipei_house.csv')

# 新增「新成屋」欄位（屋齡 < 2 年）
df['新成屋'] = (df['屋齡'] < 2).astype(int)

# 特徵欄位
features = ['行政區', '建物總面積', '屋齡', '樓層', '總樓層',
            '房數', '廳數', '衛數', '電梯', '車位類別', '新成屋']
target = '總價'

# 分割資料
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 類別欄位編碼
categorical_features = ['行政區', '車位類別']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# 建立模型
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model.fit(X_train, y_train)

# 預測函數（含新成屋）
def predict_price(行政區, 建物總面積, 屋齡, 樓層, 總樓層, 房數, 廳數, 衛數, 電梯, 車位類別):
    新成屋 = 1 if 屋齡 < 2 else 0
    input_data = pd.DataFrame([{
        '行政區': 行政區,
        '建物總面積': 建物總面積,
        '屋齡': 屋齡,
        '樓層': 樓層,
        '總樓層': 總樓層,
        '房數': 房數,
        '廳數': 廳數,
        '衛數': 衛數,
        '電梯': 電梯,
        '車位類別': 車位類別,
        '新成屋': 新成屋
    }])
    return round(model.predict(input_data)[0], 2)

# 測試
predict_price('文山區', 100.0, 1.5, 5, 10, 3, 2, 2, 1, '坡道平面')  # 預測新成屋
