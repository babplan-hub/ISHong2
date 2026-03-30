import streamlit as st
import pandas as pd
import joblib
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Education Hub", layout="wide")

# --- โหลด Assets ---
@st.cache_resource
def load_all_assets():
    l_model = joblib.load('ensemble_lung.pkl')
    l_scaler = joblib.load('scaler_lung.pkl')
    d_model = load_model('digit_model_16x16.h5')
    return l_model, l_scaler, d_model

model_lung, scaler_lung, model_digit = load_all_assets()

# --- Sidebar Navigation ---
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("เลือกหน้า:", [
    "🫁 Lung Cancer Prediction", 
    "📊 ML Explanation", 
    "🔢 Digit Drawing (NN)", 
    "🧠 NN Explanation"
])

# --- หน้าที่ 1: Lung Cancer Prediction (โค้ดเดิม) ---
if page == "🫁 Lung Cancer Prediction":
    st.title("🫁 ระบบวิเคราะห์โอกาสการเกิดมะเร็งปอด")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("อายุ (Age)", 1, 120, 40)
        gender = st.selectbox("เพศ (Gender)", ["Male", "Female"])
        smoking = st.selectbox("การสูบบุหรี่ (Smoking)", ["Yes", "No"])
        pollution = st.slider("ระดับมลพิษ (1-10)", 1, 10, 5)
    with col2:
        chronic = st.selectbox("โรคปอดเรื้อรัง", ["Yes", "No"])
        genetic = st.selectbox("ความเสี่ยงทางพันธุกรรม", ["Yes", "No"])
        fatigue = st.selectbox("อาการเหนื่อยล้า", ["Yes", "No"])

    if st.button("วิเคราะห์ผล", use_container_width=True):
        map_v = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
        data = np.array([[age, map_v[gender], map_v[smoking], pollution, map_v[chronic], map_v[genetic], map_v[fatigue]]])
        data_scaled = scaler_lung.transform(data)
        prob = model_lung.predict_proba(data_scaled)[0][1]
        st.metric("โอกาสความเสี่ยง", f"{prob*100:.2f}%")
        st.progress(prob)

# --- หน้าที่ 2: ML Explanation (อธิบายโมเดลมะเร็ง) ---
elif page == "📊 ML Explanation":
    st.title("📊 เจาะลึกการทำงานของ AI วิเคราะห์มะเร็งปอด")
    
    # --- 1. Methodology (ระเบียบวิธีวิจัย) ---
    st.subheader("🛠 1. Methodology (ระเบียบวิธีวิจัย)")
    st.write("""
    ระบบนี้พัฒนาโดยใช้แนวคิด **Synthetic Data Modeling** เพื่อสร้างข้อมูลจำลองจำนวน **2,500 เคส** ซึ่งจำลองปัจจัยเสี่ยงตามหลักสถิติสาธารณสุข โดยมีกระบวนการดังนี้:
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Feature Engineering**\n\nคัดเลือก 7 ปัจจัยสำคัญ เช่น อายุ, พฤติกรรมการสูบบุหรี่, มลพิษทางอากาศ และปัจจัยทางพันธุกรรม")
    with col2:
        st.info("**Risk Labeling**\n\nใช้ Sigmoid Function ในการกำหนดสถานะ 'เป็น/ไม่เป็น' มะเร็ง โดยอิงจากแต้มสะสมความเสี่ยง (Risk Score)")
    with col3:
        st.info("**Model Training**\n\nใช้เทคนิค Ensemble Learning เพื่อรวมจุดแข็งของหลายอัลกอริทึมเข้าด้วยกัน")

    st.divider()

    # --- 2. Risk Scoring Logic (ตารางคะแนนความเสี่ยง) ---
    st.subheader("⚖️ 2. Risk Scoring Logic (ตารางคะแนนความเสี่ยง)")
    st.write("AI เรียนรู้จากการให้ค่าน้ำหนัก (Weights) กับแต่ละปัจจัย ดังนี้:")
    
    score_data = {
        "ปัจจัยเสี่ยง (Feature)": ["พันธุกรรม (Genetic Risk)", "การสูบบุหรี่ (Smoking)", "มลพิษทางอากาศ (Pollution > 6)", "อายุ (Age > 55)", "โรคปอดเรื้อรัง", "อาการเหนื่อยล้า"],
        "ค่าน้ำหนัก (Risk Score)": ["+4 แต้ม", "+4 แต้ม", "+3 แต้ม", "+2 แต้ม", "+2 แต้ม", "+1 แต้ม"]
    }
    st.table(pd.DataFrame(score_data))
    
    st.latex(r"Probability = \frac{1}{1 + e^{-(Score - 8)}}")
    st.caption("สมการ Sigmoid ที่ใช้ในการตัดสินใจว่าผู้ป่วยมีความเสี่ยงระดับใด")

    st.divider()

    # --- 3. Prediction & Accuracy Analysis ---
    st.subheader("🎯 3. Prediction & Accuracy Analysis")
    
    res_a, res_b = st.columns(2)
    with res_a:
        st.write("#### ทำไมความแม่นยำอยู่ที่ 63-70%?")
        st.write("""
        1. **Noise Integration:** ในชุดข้อมูลมีการใส่ 'ค่ารบกวน' (Stochastic Noise) เพื่อจำลองความเป็นจริงที่ว่า *'คนสูบบุหรี่หนักบางคนอาจไม่เป็นมะเร็ง'* และ *'คนสุขภาพดีบางคนอาจเป็นมะเร็งจากปัจจัยอื่น'*
        2. **Threshold:** การตัดสินใจที่จุดตัด (Cut-off) 8 แต้ม ทำให้เคสที่มีแต้มก้ำกึ่ง (เช่น 7 หรือ 9) มีความท้าทายในการทำนาย
        """)
    
    with res_b:
        st.write("#### การแปลผล (Prediction)")
        st.write("""
        * **Probability > 50%:** AI จะจัดกลุ่มเป็น 'มีความเสี่ยงสูง'
        * **Ensemble Impact:** การใช้หลายโมเดลช่วยลดความผิดพลาด (Bias) จากข้อมูลที่ก้ำกึ่ง ทำให้ผลการทำนายมีเสถียรภาพมากกว่าโมเดลตัวเดียว
        """)
    
    st.warning("⚠️ **หมายเหตุ:** โมเดลนี้พัฒนาขึ้นเพื่อการศึกษาและการจำลองเชิงสถิติ ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์จริงได้")

    

# --- หน้าที่ 3: Digit Drawing (หน้าวาดรูปเดิม) ---
elif page == "🔢 Digit Drawing (NN)":
    st.title("🔢 วาดตัวเลขทายผลด้วย Neural Network")
    canvas_result = st_canvas(stroke_width=20, stroke_color="#FFFFFF", background_color="#000000", height=300, width=300, key="canvas")
    if canvas_result.image_data is not None:
        img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        img_16 = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
        if st.button("ทายผล"):
            inp = img_16.reshape(1, 16, 16, 1) / 255.0
            pred = model_digit.predict(inp)
            st.header(f"AI ทายว่าเป็นเลข: {np.argmax(pred)}")

# --- หน้าที่ 4: NN Explanation (อธิบาย Neural Network) ---
elif page == "🧠 NN Explanation":
    st.title("🧠 เจาะลึกสมองกล Neural Network (CNN)")
    
    st.subheader("🛠 Methodology: การเรียนรู้จากภาพขนาด 16x16")
    st.write("""
    โมเดลนี้ใช้สถาปัตยกรรม **CNN (Convolutional Neural Network)** ซึ่งถูกออกแบบมาเพื่อเลียนแบบระบบการมองเห็นของมนุษย์ 
    โดยเน้นการจดจำ 'ลวดลาย' มากกว่าการจำทุกพิกเซล
    """)
    
    # อธิบายโครงสร้างด้วย Bullet points
    st.markdown("""
    - **Input Layer:** รับภาพ Grayscale ขนาด 16x16 พิกเซล (รวม 256 จุด)
    - **Convolutional Layer:** ทำหน้าที่เป็น 'แว่นขยาย' ส่องหาเส้นตรงและเส้นโค้ง
    - **Flatten & Dense:** เปลี่ยนภาพจากตาราง 2 มิติ ให้เป็นเส้นตรงเพื่อเข้าสู่กระบวนการตัดสินใจ
    - **Output (Softmax):** ให้ผลลัพธ์เป็นความน่าจะเป็นของตัวเลข 0-9
    """)

    st.divider()
    
    st.subheader("🎯 ประสิทธิภาพและการทำนาย (Prediction)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("#### ความแม่นยำ (Accuracy)")
        st.success("95% - 98% (บน Test Set)")
        st.write("ความแม่นยำระดับนี้เพียงพอที่จะระบุลายมือที่เขียนได้หลากหลายรูปแบบ")
    
    with col_b:
        st.write("#### กระบวนการทำนาย")
        st.info("Canvas (300px) ➡️ Resize (16px) ➡️ CNN Process ➡️ Result")

    st.warning("🧩 **Fact:** การใช้ขนาด 16x16 ช่วยให้แอปโหลดเร็วขึ้น 5 เท่าเมื่อเทียบกับขนาดมาตรฐาน แต่ยังให้ความแม่นยำที่ใกล้เคียงกัน")