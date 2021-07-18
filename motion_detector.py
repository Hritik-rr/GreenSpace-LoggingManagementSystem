import cv2, time
import pandas as pd
from datetime import datetime
first_frame=None
status_list=[None,None]
times=[]
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
df=pd.DataFrame(columns=["Start","End"])


while True:

    check, frame=video.read()
    status=0
    Gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    Gray=cv2.GaussianBlur(Gray,(21,21),0)
    if first_frame is None:
        first_frame=Gray
        continue
    
    delta_frame=cv2.absdiff(first_frame,Gray)
    thresh_frame=cv2.threshold(delta_frame,90,255,cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame,None, iterations=2)

    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for count in cnts:
        if cv2.contourArea(count) < 10000:
            continue
        
        status=1
        (x,y,w,h)=cv2.boundingRect(count)
        cv2.rectangle(frame,(x,y),(x+w,x+h),(0,255,0),5)
 
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())    

    cv2.imshow("Capturing", Gray)
    cv2.imshow("Delta",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("color Frame",frame)
    key=cv2.waitKey(1)
    
    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
    
print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)


df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows