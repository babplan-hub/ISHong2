import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras import layers, models
import sys
sys.stdout.reconfigure(encoding='utf-8')

def load_mnist_ubyte(images_path, labels_path):
    # ฟังก์ชันสำหรับแกะไฟล์ binary .ubyte
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28)
    
    return images, labels

# --- 1. ระบุชื่อไฟล์ ubyte ของคุaณตรงนี้ (แก้ให้ตรงกับที่โหลดมา) ---
# ปกติชื่อจะประมาณ: train-images-idx3-ubyte และ train-labels-idx1-ubyte
try:
    x_train, y_train = load_mnist_ubyte('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
    print("แกะไฟล์ ubyte สำเร็จ!")
except FileNotFoundError:
    print("❌ ไม่พบไฟล์ ubyte กรุณาเช็คชื่อไฟล์ในโฟลเดอร์อีกครั้ง")
    exit()

# --- 2. เตรียมข้อมูล (Resize เป็น 16x16) ---
x_train = np.expand_dims(x_train, axis=-1).astype('float32')
x_train_16 = tf.image.resize(x_train, [16, 16]).numpy() / 255.0

# --- 3. สร้างและเทรนโมเดล ---
model = models.Sequential([
    layers.Input(shape=(16, 16, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("กำลังเริ่มเทรน...")
model.fit(x_train_16, y_train, epochs=5, batch_size=32)

# --- 4. เซฟเป็นไฟล์ .h5 เพื่อเอาไปใช้ใน Streamlit ---
model.save('digit_model_16x16.h5')
print("✅ สำเร็จ! ได้ไฟล์ 'digit_model_16x16.h5' มาแล้ว")