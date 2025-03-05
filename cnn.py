import cv2
import numpy as np
from keras.models import load_model

# Load model yang telah dilatih
model_path = 'Intelligent-Control-Week3/cnn_model.h5'
try:
    model = load_model(model_path)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Label kelas yang sesuai dengan model
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame!")
        break

    # Mode Night Vision dengan konversi ke skala abu-abu
    night_vision = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    night_vision = cv2.applyColorMap(night_vision, cv2.COLORMAP_JET)

    # Preprocessing gambar
    img = cv2.resize(frame, (150, 150))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediksi kelas
    pred = model.predict(img)
    max_index = np.argmax(pred)  # Indeks kelas dengan probabilitas tertinggi
    label = class_labels[max_index]  # Ambil label kelas
    confidence = pred[0][max_index] * 100  # Ambil nilai probabilitas (dalam persen)

    # Tampilkan hasil
    text = f'Class: {label} ({confidence:.2f}%)'
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    cv2.imshow('Night Vision', night_vision)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
