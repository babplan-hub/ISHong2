import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# โหลด dataset ที่สร้าง
df = pd.read_csv('lung_cancer_2500_cases.csv')

# map ค่า
map_v = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
df = df.replace(map_v)

X = df.drop("Lung_Cancer", axis=1)
y = df["Lung_Cancer"]

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train
model = RandomForestClassifier()
model.fit(X_scaled, y)

# save
joblib.dump(model, "ensemble_lung.pkl")
joblib.dump(scaler, "scaler_lung.pkl")

print("✅ train + save model สำเร็จ")