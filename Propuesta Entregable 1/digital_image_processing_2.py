import cv2

#detección de rostros
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#captura señal de video
capture = cv2.VideoCapture('video.wmv')

#captura la camara
camara = cv2.VideoCapture(0)

while(True):
    #captura frame
    ret, frame = capture.read()

    #Rotación de imagen
    rows, cols = frame.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    frame = cv2.warpAffine(frame, M, (cols, rows))

    #Se convierte el frame a gris
    frameGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostro
    faces = face_cascade.detectMultiScale(frameGris, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frameGris, (x, y), (x + w, y + h), (255, 0, 0), 2)

        resolucion = (w + h) / 8

        # Cálculo de posición de ojos usando Ecuaciones
        # Ojo derecho
        cv2.circle(frameGris, (int(x + w * 0.3), int(y + h * 0.4)), int(resolucion * 0.3), (255, 255, 255), 2)
        # Ojo izquierdo
        cv2.circle(frameGris, (int(x + w * 0.7), int(y + h * 0.4)), int(resolucion * 0.3), (255, 255, 255), 2)

        # Cálculo de posición de las orejas usando Ecuaciones
        # Oreja derecha
        cv2.circle(frameGris, (int(x + w * 0.05), int(y + h * 0.45)), int(resolucion * 0.3), (20, 20, 140), 2)
        # Oreja izquierda
        cv2.circle(frameGris, (int(x + w * 0.95), int(y + h * 0.45)), int(resolucion * 0.3), (20, 20, 140), 2)

        # Cálculo de posición de boca usando Ecuaciones
        # arco inferior de boca
        cv2.ellipse(frameGris, (int(x + w * 0.5), int(y + h * 0.65)), (int(resolucion), int(resolucion)), 45, 10, 80, (0, 0, 255), 0)
        # arco superior de boca
        cv2.ellipse(frameGris, (int(x + w * 0.5), int(y + h * 1.03)), (int(resolucion), int(resolucion)), 225, 10, 80, (0, 0, 255), 0)

        # Cálculo de posición de nariz usando Ecuaciones
        cv2.rectangle(frameGris, (int(x + w * 0.45), int(y + h * 0.4)), (int(x + w * 0.55), int(y + h * 0.62)), (0, 0, 0), 2)

    # Muestra el video resultante con el rostro y sus características detectadas
    cv2.imshow("CAPTURE", frameGris)


    # camara frame
    ret2, frame2 = camara.read()

    # Se convierte el frame a gris
    frameGris2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detección de rostro
    faces2 = face_cascade.detectMultiScale(frameGris2, 1.1, 5)

    for (x, y, w, h) in faces2:
        cv2.rectangle(frameGris2, (x, y), (x + w, y + h), (255, 0, 0), 2)

        resolucion = (w + h) / 8

        # Cálculo de posición de ojos usando Ecuaciones
        # Ojo derecho
        cv2.circle(frameGris2, (int(x + w * 0.3), int(y + h * 0.4)), int(resolucion * 0.3), (255, 255, 255), 2)
        # Ojo izquierdo
        cv2.circle(frameGris2, (int(x + w * 0.7), int(y + h * 0.4)), int(resolucion * 0.3), (255, 255, 255), 2)

        # Cálculo de posición de las orejas usando Ecuaciones
        # Oreja derecha
        cv2.circle(frameGris2, (int(x + w * 0.05), int(y + h * 0.45)), int(resolucion * 0.3), (20, 20, 140), 2)
        # Oreja izquierda
        cv2.circle(frameGris2, (int(x + w * 0.95), int(y + h * 0.45)), int(resolucion * 0.3), (20, 20, 140), 2)

        # Cálculo de posición de boca usando Ecuaciones
        # arco inferior de boca
        cv2.ellipse(frameGris2, (int(x + w * 0.5), int(y + h * 0.65)), (int(resolucion), int(resolucion)), 45, 10, 80, (0, 0, 255), 0)
        # arco superior de boca
        cv2.ellipse(frameGris2, (int(x + w * 0.5), int(y + h * 1.03)), (int(resolucion), int(resolucion)), 225, 10, 80, (0, 0, 255), 0)

        # Cálculo de posición de nariz usando Ecuaciones
        cv2.rectangle(frameGris2, (int(x + w * 0.45), int(y + h * 0.4)), (int(x + w * 0.55), int(y + h * 0.62)), (0, 0, 0), 2)

    # Muestra el video resultante con el rostro y sus características detectadas
    cv2.imshow("CAMARA", frameGris2)

    key = cv2.waitKey(1)  # Retraso en milisegundos para leer el siguiente frame

    if (key == 27):  # Termina presionando la tecla Esc
        break

capture.release()
camara.release()
cv2.destroyAllWindows()