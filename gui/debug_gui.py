import cv2
import tkinter as tk

print("== DEBUG GUI ==")

# 1. PROBAR TKINTER
print("Probando ventana...")
root = tk.Tk()
root.title("Debug TK")
root.geometry("300x100")

label = tk.Label(root, text="Tkinter funciona!")
label.pack()

root.update()
print("Ventana creada correctamente.")

# 2. PROBAR CAMARA
print("Probando cámara...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cámara NO se pudo abrir.")
else:
    print("Cámara abierta correctamente.")

cap.release()

print("Todo OK. Manteniendo ventana…")
root.mainloop()
