import numpy as np
import cv2

#Archivos de haarcascade para caracteristicas de...
#detección de rostros
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#detección de ojos
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#detección de boca
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
#detección de nariz
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

#captura señal de video
cap = cv2.VideoCapture('video.wmv')

while(True):
    #captura frame
    ret, frame = cap.read()
    
    #Rotación de imagen
    rows, cols = frame.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv2.warpAffine(frame, M, (cols, rows))
    
    #Se convierte el frame a gris
    frameGris = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    #Deteccion de rostro
    faces = face_cascade.detectMultiScale(frameGris, 1.1, 5)

    for (x,y,w,h) in faces:     #x,y posición TL del box del rostro, w es ancho, h es alto
        cv2.rectangle(dst,(x,y),(x+w,y+h),(255,0,0),2)    #dibuja rectangulo del rostro detectado - color Azul

        #Extrae ROI del rostro detectado
        roi_gray = frameGris[y:y+h, x:x+w]
        roi_color = dst[y:y+h, x:x+w]

        #Detección de ojos a partir de ROI extraido
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:  #ex,ey posición TL del box del ojo, ew es ancho, eh es alto
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  #dibuja rectangulo del ojo detectado  - color Verde

        #Detección de boca a partir de ROI extraido
        bocas = mouth_cascade.detectMultiScale(roi_gray)

        for (mx,my,mw,mh) in bocas:
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)  #dibuja rectangulo de boca detectada  - color Rojo

        #Deteccion de nariz a partir de ROI extraido
        narices = nose_cascade.detectMultiScale(roi_gray)

        for (nx,ny,nw,nh) in narices:
            cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,255,255),2)  #dibuja rectangulo de nariz detectada  - color Blanco

    #Se muestra el frame resultante
    cv2.imshow('frame',dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


