#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model


# In[2]:


model = load_model("mask_model.h5")


# In[3]:


import cv2


# In[ ]:


video = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret,frame = video.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray)
        for face in faces:
            x,y,w,h = face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            crop_frame = frame[y:y+h, x:x+w]
            crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            crop_frame = cv2.resize(crop_frame, (100,200))
            shape = crop_frame.shape
            
            crop_frame = crop_frame/255
            pred_frame = crop_frame.reshape(-1, shape[0]*shape[1])
#             print(crop_frame, crop_frame.shape)
            pred = model.predict(pred_frame)
            print(pred[0][0])
            
            if pred[0][0] >= 0.7:
                cv2.putText(frame,"Masks are ON",(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
                print("Masks are ON")
            
                
            if pred[0][0] < 0.7:
                print("Masks are OFF")
                cv2.putText(frame,"Masks are OFF",(10,30), font, 1,(255,255,255),1,cv2.LINE_AA)
            
        cv2.imshow("Frame", frame)
                
    if cv2.waitKey(1) & 0xFF == 'q':
        break
        
video.release()
video.destroyAllWindows()
            
            

    


# In[ ]:




