import cv2

def prom(list):
    return sum(list) / len(list)

# Classifier XML
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Captura la señal desde un archivo de video
capture = cv2.VideoCapture("faces.wmv")
# Se definen listas para las coordenadas [i, j]
i = []
j = []
# Se definen listas para las coordenadas [r, c]
r = []
c = []

while(True):
    # Lectura frame por frame
    ret, frame = capture.read()

    # Rotacion
    rows, cols, canales1 = frame.shape
    temp = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
    frame = cv2.warpAffine(frame, temp, (cols, rows))
    
    # Convierte el frame de BGR a GRIS
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostro
    faces = face_cascade.detectMultiScale(frame_gray, 1.1, 5)

    # Se dibuja el rectangulo de visión
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_gray,'Rango de vision',(100,470), font, 0.5,(0,255,0),1,cv2.LINE_AA)
    cv2.rectangle(frame_gray,(100,20),(530,456),(0,255,0),2)


    for (x, y, w, h) in faces:
        # Se dibuja el rectangulo para cada rostro
        cv2.rectangle(frame_gray, (x, y), (x + w, y + h), (255, 255, 255), 2)
        # Se dibuja el circulo central para cada rostro
        cv2.circle(frame_gray, (int(x + w / 2), int(y + h / 2)), 2, (255, 255, 255), 2)
        
        i.append(int(x + w / 2))
        j.append(int(y + h / 2))

    if (len(i) != 0):
        cv2.circle(frame_gray, (int(prom(i)), int(prom(j))), 2, (0, 255, 0), 2)
        # Se guardan las coordenadas como historial
        r.append(int(prom(i)))
        c.append(int(prom(j)))
        print ("---------------------------------")
        print ("Coordenada en el eje X: ", prom(i))
        print ("Coordenada en el eje Y: ", prom(j))

    # Se vacian las listas de coordenadas para cada frame
    i.clear()
    j.clear()
		
    # Enviar trama
    # Enviar trama
    
    # Muestra el video resultante con el rostro y sus características detectadas
    cv2.imshow("Web Cam", frame_gray)

    # Retraso en milisegundos para leer el siguiente frame
    key = cv2.waitKey(1)  
    # Termina presionando la tecla esc
    if (key == 27):  
        break

capture.release()
cv2.destroyAllWindows()
