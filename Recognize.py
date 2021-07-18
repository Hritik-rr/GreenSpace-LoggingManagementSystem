import cv2
import pickle
import csv
import time
import datetime
import pandas as pd
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
col_names =  ['Id','Name','Date','Time']
attendance = pd.DataFrame(columns = col_names) 


lables={}
with open("labels.pickle","rb") as f:
    og_labels = pickle.load(f)
    labels={v:k for k,v in og_labels.items()}


video_capture = cv2.VideoCapture(0)


while True:

    ret, frame = video_capture.read()#takes images frame by frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(#gives a list of coordinates for image detected
        gray,
        scaleFactor=1.2,#Parameter specifying how much the image size is reduced at each image scale.
        minNeighbors=5,#Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        minSize=(40, 40)#Minimum possible object size. Objects smaller than that are ignored.
    )

    #frame=cv2.flip(frame,1)
    for (x, y, w, h) in faces:
        #print(faces)
        #print(len(faces))
        roi_gray=gray[y:y+h,x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        id_, conf= recognizer.predict(roi_gray)
        if conf<50 :
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=1
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            ts = time.time()      
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            attendance.loc[len(attendance)] = [id_,name,date,timeStamp]
                
        elif conf>75:
            noOfFile=len(os.listdir("ImagesUnknown"))+1
            cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", frame[y:y+h,x:x+w])
        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')

    cv2.imshow('Video',frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+".csv"
    attendance.to_csv(fileName,index=False)

video_capture.release()

cv2.destroyAllWindows()