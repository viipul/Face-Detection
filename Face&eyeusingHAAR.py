import cv2
import numpy as np
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(face_classifier)
image=cv2.imread('image.jpeg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#convert image to grayscale
cv2.imshow('grey',gray)
cv2.waitKey(5000)
print(gray.shape)

faces=face_classifier.detectMultiScale(gray,1.3,5)
#detectMultiScale detects features of the pae and store in faces
#it will give 4 values
#that are x,y coordinates widht and height of the face
print(faces)
if faces is ():
    print('NO FACES FOUND ')
#WE ITERATE THROUGH FACES AND DRAW RECTANGLE OVER EACH FACE IN FACES
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('Face detection',image)
    cv2.waitKey(0)
cv2.destroyAllWindows()

#eye classifiaction
eye_classifier=cv2.CascadeClassifier('Haarcascades\\haarcascade_eye.xml')
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),5)
    cv2.imshow('Face detection',image)
    cv2.waitKey(0)
    roi_gray=gray[y:y+h,x:x+h]
    roi_col=image[y:y+h,x:x+h]
    eyes=eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_col,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)
        cv2.imshow('Eye Detection',roi_col)
        cv2.waitKey(0)
cv2.destroyAllWindows()
