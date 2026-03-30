import pandas as pd
import numpy as np

# ตั้งค่าจำนวนข้อมูลที่ต้องการ
n_samples = 2500
np.random.seed(42)

# สร้างข้อมูลจำลองตามสถิติและความสัมพันธ์ของตัวแปร
data = {
    'Age': np.random.randint(18, 90, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Smoking': np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6]),
    'Air_Pollution_Level': np.random.randint(1, 11, n_samples),
    'Chronic_Lung_Disease': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    'Genetic_Risk': np.random.choice(['Yes', 'No'], n_samples, p=[0.25, 0.75]),
    'Fatigue': np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55])
}

df = pd.DataFrame(data)

# ฟังก์ชันคำนวณโอกาสเป็นมะเร็ง (Logic สำหรับสร้าง Label ให้ AI เรียนรู้)
def calculate_cancer(row):
    score = 0
    if row['Age'] > 55: score += 2
    if row['Smoking'] == 'Yes': score += 4
    if row['Air_Pollution_Level'] > 6: score += 3
    if row['Chronic_Lung_Disease'] == 'Yes': score += 2
    if row['Genetic_Risk'] == 'Yes': score += 4
    if row['Fatigue'] == 'Yes': score += 1
    
    # ถ้าแต้มรวมสูงกว่า 8 หรือมีการสุ่มเล็กน้อย (Noise) เพื่อให้ AI ไม่ทายง่ายเกินไป
    probability = 1 / (1 + np.exp(-(score - 8))) # Sigmoid function
    return 1 if np.random.random() < probability else 0

df['Lung_Cancer'] = df.apply(calculate_cancer, axis=1)

# บันทึกเป็นไฟล์ CSV
df.to_csv('lung_cancer_2500_cases.csv', index=False)
print("lung_cancer_2500_cases.csv 2500!")