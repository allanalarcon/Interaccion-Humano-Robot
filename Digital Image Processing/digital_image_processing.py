import cv2

# Classifier XML
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Captura la señal de video desde la camara
capture = cv2.VideoCapture(0)

while(True):
    # Lectura frame por frame
    ret, frame = capture.read()
    # Convierte el frame de BGR a GRIS
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostro
    faces = face_cascade.detectMultiScale(frame_gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame_gray, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.circle(frame_gray, (int(x + w / 2), int(y + h / 2)), 2, (255, 255, 255), 2)
		
    # Muestra el video resultante con el rostro y sus características detectadas
    cv2.imshow("Web Cam", frame_gray)

    # Retraso en milisegundos para leer el siguiente frame
    key = cv2.waitKey(1)  
    # Termina presionando la tecla Esc
    if (key == 27):  
        break

capture.release()
cv2.destroyAllWindows()
