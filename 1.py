import cv2
import numpy as np


face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
Eye_detector1 = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")


emoji_face = cv2.imread('images.png',0)
emoji_eye = cv2.imread('eye.png',0)
emoji_mouth = cv2.imread('smile.png',0)



video_cap = cv2.VideoCapture(0)
flag_face = False

while (True):

    ret, frame = video_cap.read()
    

    if ret == False:
        break



  
    cv2.imshow('frame', frame)
    

    faces = face_detector.detectMultiScale(frame, 1.3)
    eyes = Eye_detector1.detectMultiScale(frame, 1.2)
    mouth = smile_detector.detectMultiScale(frame, 1.2)

    if cv2.waitKey(1) & 0xFF == ord('1'):

        
        for (x, y, w, h) in faces:
            emoji_on_face = cv2.resize(emoji_face, (w, h))
            frame[y:y+h,x:x+w]=emoji_on_face
            for i in range(y,y+h):
                for j in range(x,x+w):
                    if frame[i,j]>=246 and frame[i,j]<=255:
                        frame[i,j]=[i,j]
   

    
    if cv2.waitKey(1) & 0xFF == ord('2'):
        for (ex, ey, ew, eh) in eyes:
            emoji_eyes = cv2.resize(emoji_eye, (ew, eh))
            frame[ey:ey+eh, ex:ex+ew] = emoji_eyes
            for i in range(ey,ey+eh):
                for j in range(ex,ex+ew):
                    if frame[i,j]==0:
                        frame[i,j]=[i,j]


 

        for (mx, my, mw, mh) in mouth:
            emoji_smile = cv2.resize(emoji_mouth, (mw, mh))
            frame[my:my + mh, mx:mx + mw] = emoji_smile
            for i in range(my,my+mh):
                for j in range(mx,mx+mw):
                    if frame[i,j]==255:
                       frame[i,j]=[i,j]

        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('3'):
        for (x, y, w, h) in faces:
           temp=cv2.resize(frame[y:y+h,x:x+w],(20,20),interpolation=cv2.INTER_LINEAR)
           frame[y:y+h,x:x+w]=cv2.resize(temp,(w,h),interpolation=cv2.INTER_NEAREST)   

        cv2.imshow('frame',frame)      

    if cv2.waitKey(1) & 0xFF == ord('4'):
        for (x,y,w,h) in faces:
            frame[y:y+h,x:x+w]= cv2.rotate(frame[y:y+h,x:x+w],cv2.ROTATE_90_CLOCKWISE)

        cv2.imshow('frame',frame)    

    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

video_cap.release()
cv2.destroyAllWindows()