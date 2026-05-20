# %% [markdown]
# # Mô hình Dự đoán Khách hàng rời bỏ (Customer Churn)
# File này được thiết kế theo dạng Interactive Python (Jupyter-like). 
# Nếu bạn mở file này trong VS Code, bạn có thể click "Run Cell" phía trên mỗi khối lệnh (được đánh dấu bằng `# %%`) để chạy code y hệt như Google Colab.

# %%
# 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import joblib

# %%
# 2. ĐỌC VÀ LÀM SẠCH DỮ LIỆU
print("Đang đọc dữ liệu...")
df = pd.read_csv("Telco-Customer-Churn.csv")

# Xử lý TotalCharges bị thiếu
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Bỏ cột customerID vì không có giá trị dự đoán
df = df.drop('customerID', axis=1)

# Chuyển đổi Target (Churn) thành 1 và 0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Tách Features (X) và Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Phân chia Train / Test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm thử: {X_test.shape}")

# %%
# 3. TIỀN XỬ LÝ (PREPROCESSING PIPELINE)
# Xác định các cột số (numeric) và phân loại (categorical)
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if X[col].dtype == 'object' and col not in numeric_features]

# Tạo Pipeline biến đổi dữ liệu
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Áp dụng Pipeline lên tập Train và Test
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("Đã hoàn thành tiền xử lý (Scaling và One-Hot Encoding).")

# %%
# 4. XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU BẰNG SMOTE
print("Đang cân bằng dữ liệu bằng SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
print(f"Phân phối Churn sau SMOTE: \n{pd.Series(y_train_resampled).value_counts()}")

# %%
# 5. HUẤN LUYỆN MÔ HÌNH CATBOOST (STATE-OF-THE-ART)
print("\nĐang huấn luyện mô hình CatBoost...")
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    verbose=100,
    random_seed=42
)

# Đưa dữ liệu vào huấn luyện
model.fit(X_train_resampled, y_train_resampled, eval_set=(X_test_processed, y_test), early_stopping_rounds=50)

# %%
# 6. ĐÁNH GIÁ MÔ HÌNH
y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

print("\n--- BÁO CÁO KẾT QUẢ ĐÁNH GIÁ ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# %%
# 7. LƯU MÔ HÌNH & PIPELINE ĐỂ DÙNG CHO WEB APP
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(model, 'catboost_model.pkl')
print("\nĐã lưu preprocessor.pkl và catboost_model.pkl thành công!")
