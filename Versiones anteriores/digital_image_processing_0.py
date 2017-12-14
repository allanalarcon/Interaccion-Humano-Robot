import cv2

video = cv2.VideoCapture("Video_Prueba.mp4")

camara = cv2.VideoCapture(0)

#Pantallas para el video prueba
cv2.namedWindow("Video Original:", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Video Volteado:", cv2.WINDOW_AUTOSIZE)

#Pantallas para la cámara
cv2.namedWindow("Camara Original:", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Camara Volteado:", cv2.WINDOW_AUTOSIZE)

while(True):

#video prueba
    ret, frame = video.read()

    if(not ret):
        break

    #rotando video
    rows, cols, canales1 = frame.shape
    temp = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
    frameV = cv2.warpAffine(frame, temp, (cols, rows))

    cv2.imshow("Video Original:", frame)
    cv2.imshow("Video Volteado:", frameV)

#camara
    ret2, frame2 = camara.read()

    if(not ret2):
        break

    #rotando camara
    rows2, cols2, canales2 = frame2.shape
    temp2 = cv2.getRotationMatrix2D((cols2/2, rows2/2), 90, 1)
    frame2V = cv2.warpAffine(frame2, temp2, (cols2, rows2))

    cv2.imshow("Camara Original:", frame2)
    cv2.imshow("Camara Volteado:", frame2V)

    key = cv2.waitKey(1)

    if (key==27):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

video.release()
camara.release()