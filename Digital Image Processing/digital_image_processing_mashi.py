import cv2
import numpy as np

# Coordenadas de los centroides de cada rostro y el área
i = []
j = []
area = []
areaTemp = []

# Coordenadas del punto a seguir
r = []
c = []

# Classifier XML
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Captura la señal desde un archivo de video
capture = cv2.VideoCapture("faces.wmv")

# Lectura de un frame para hacer el calculo de rotación
ret, frame = capture.read()

# Rotación
rows, cols, canales1 = frame.shape
temp = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
cos = np.abs(temp[0, 0])
sin = np.abs(temp[0, 1])
nuevoW = int((rows * sin) + (cols * cos))
nuevoH = int((rows * cos) + (cols * sin))
temp[0, 2] += (nuevoW / 2) - (cols / 2)
temp[1, 2] += (nuevoH / 2) - (rows / 2)

while (True):
    # Lectura del frame
    ret, frame = capture.read()

    # Rotación
    frame = cv2.warpAffine(frame, temp, (nuevoW, nuevoH))

    # Convierte el frame de BGR a GRIS
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostro
    faces = face_cascade.detectMultiScale(frame_gray, 1.1, 5)

    # Se dibuja el rectangulo de visión
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_gray, 'Area de vision', (200, 230), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(frame_gray, (200, 120), (280, 220), (0, 255, 0), 2)

    for x, y, w, h in faces:
        # Se dibuja el rectangulo para cada rostro
        cv2.rectangle(frame_gray, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Se calcula el centroide de cada rostro y el área
        xc = int(x + w / 2)
        yc = int(y + h / 2)
        cv2.circle(frame_gray, (xc, yc), 2, (255, 255, 255), 2)
        i.append(xc)
        j.append(yc)
        area.append(w * h)

        # IF EN ETAPA DE PRUEBA
        if len(r) > 1:
            if (yc - r[len(r) - 1]) > 25:
                i.pop()
                j.pop()
                area.pop()

        areaTemp = area.copy()
        areaTemp.sort(reverse=True)

    # Se valida que existan coordenadas
    if len(i) == len(j) and len(i) > 0:
        # En caso de haber una coordenada, esa será el punto a seguir
        if len(i) == 1:
            c.append(i[0])
            r.append(j[0])
        else:
            # Se usará la coordenada del rostro que esté más cerca (área más grande)
            if (areaTemp[0] - areaTemp[1]) >= 2500:
                c.append(i[area.index(areaTemp[0])])
                r.append(j[area.index(areaTemp[0])])
            # Se calculará la mediana de las coordenadas de los rostros
            else:
                iTemp = i.copy()
                jTemp = j.copy()
                i.sort()
                j.sort()
                if len(i) % 2 == 0:
                    c.append(int((i[int(len(i) / 2 - 1)] + i[int(len(i) / 2)]) / 2))
                    r.append(int((j[int(len(j) / 2 - 1)] + j[int(len(j) / 2)]) / 2))
                else:
                    c.append(int(i[int(len(i) / 2)]))
                    r.append(jTemp[iTemp.index(c[len(c) - 1])])
                jTemp.clear()
                iTemp.clear()

        # Se dibuja el punto de referencia
        cv2.circle(frame_gray, (c[len(c) - 1], r[len(r) - 1]), 2, (0, 255, 0), 2)
        print("---------------------------------")
        print("Coordenada en el eje X: ", c[len(c) - 1])
        print("Coordenada en el eje Y: ", r[len(r) - 1])

    # Se vacian las listas
    i.clear()
    j.clear()
    area.clear()
    areaTemp.clear()

    # Muestra el vídeo
    cv2.imshow("Web Cam", frame_gray)

    # Retraso en milisegundos para leer el siguiente frame
    key = cv2.waitKey(1)

    # Termina presionando la tecla esc
    import time

    if key == 27:
        time.sleep(50)

capture.release()
cv2.destroyAllWindows()
