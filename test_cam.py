import cv2

print("Buscando cámaras...")
for i in range(0, 6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Cámara encontrada en índice: {i}")
        cap.release()
    else:
        print(f"❌ Nada en índice: {i}")
