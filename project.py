import tkinter as tk
from tkinter import Message, Text
import cv2, os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import pickle
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
window.title("Project Exibition")
window.geometry("1280x720")
window.configure(background="white")
window.grid_rowconfigure(0, weight=1)           
window.grid_columnconfigure(0, weight=1)
message = tk.Label(
    window,
    text="Staff and Visitor Logging & Management System",
    fg="black",
    width=40,
    height=3,
    font=("Georgia",25,"bold underline")
)
message.place(x=350, y=20)
lbl2 = tk.Label(
    window,
    text="Enter Name",
    width=20,
    fg="blue",
    height=2,
    font=("Georgia", 15, " bold "),
)
lbl2.place(x=400, y=200)
txt2 = tk.Entry(window, width=20, font=("Georgia", 15, " bold "))
txt2.place(x=700, y=215)


def capture():
    name = txt2.get()
    namedir = r"Images" + "\\" + name
    os.mkdir(namedir)
    i = 1
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        check, frame = video.read()
        cv2.imshow("Capturing", cv2.flip(frame, 1))
        img_item = str(i) + ".jpg"
        os.chdir(namedir)
        cv2.imwrite(img_item, frame)
        i += 1
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
        elif i > 60:
            break
    video.release()
    cv2.destroyAllWindows()


def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "Images")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                # print(label,path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                # print(label_ids)
                # y_labels.append(labels) #some number
                # x_train.append(path)    #verify this image,turn into a numpy array
                pil_image = Image.open(path).convert("L")  # Graysacle
                image_array = np.array(pil_image, "uint8")
                # print(image_array)
                faces = faceCascade.detectMultiScale(# gives a list of coordinates for face detected
                    image_array,
                    scaleFactor=1.2,  # Parameter specifying how much the image size is reduced at each image scale.
                    minNeighbors=5,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
                )
                for (x, y, w, h) in faces:
                    roi = image_array[y : y + h, x : x + w]
                    x_train.append(roi)
                    y_labels.append(id_)
                    # print(y_labels)
    # print(y_labels)
    # print(x_train)
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")


def Track():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    col_names = ["Id", "Name", "Date", "Time"]
    attendance = pd.DataFrame(columns=col_names)

    lables = {}
    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    video_capture = cv2.VideoCapture(0)

    while True:

        ret, frame = video_capture.read()  # takes images frame by frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(  # gives a list of coordinates for image detected
            gray,
            scaleFactor=1.2,  # Parameter specifying how much the image size is reduced at each image scale.
            minNeighbors=5,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            minSize=(
                40,
                40,
            ),  # Minimum possible object size. Objects smaller than that are ignored.
        )

        # frame=cv2.flip(frame,1)
        for (x, y, w, h) in faces:
            # print(faces)
            # print(len(faces))
            roi_gray = gray[y : y + h, x : x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            id_, conf = recognizer.predict(roi_gray)
            if conf < 50:
                print(id_)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 1
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                attendance.loc[len(attendance)] = [id_, name, date, timeStamp]

            elif conf > 100:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite(
                    "ImagesUnknown\Image" + str(noOfFile) + ".jpg",
                    frame[y : y + h, x : x + w],
                )

            attendance = attendance.drop_duplicates(subset=["Id"], keep="first")

        cv2.imshow("Video", frame)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        Hour, Minute, Second = timeStamp.split(":")
        fileName = "Attendance\Attendance_" + date + ".csv"
        attendance.to_csv(fileName, index=False)

    video_capture.release()

    cv2.destroyAllWindows()


takeImg = tk.Button(
    window,
    text="Capture Images",
    command=capture,
    fg="Blue",
    width=20,
    height=3,
    activebackground="cyan",
    font=("georgia", 15, " bold "),
)
takeImg.place(x=200, y=500)

trainImg = tk.Button(
    window,
    text="Train Images",
    command=train,
    fg="Blue",
    width=20,
    height=3,
    activebackground="cyan",
    font=("georgia", 15, " bold "),
)
trainImg.place(x=500, y=500)

trackImg = tk.Button(
    window,
    text="Start Logging",
    command=Track,
    fg="Blue",
    width=20,
    height=3,
    activebackground="cyan",
    font=("georgia", 15, " bold "),
)
trackImg.place(x=800, y=500)
quitWindow = tk.Button(
    window,
    text="Quit",
    command=window.destroy,
    fg="Blue",
    width=20,
    height=3,
    activebackground="cyan",
    font=("georgia", 15, " bold "),
)
quitWindow.place(x=1100, y=500)
Write = tk.Text(
    window,
    background=window.cget("background"),
    borderwidth=0,
    font=("georgia", 20, "bold"),
)
Write.tag_configure("superscript", offset=10)
Write.insert("insert", "Developed by Abhijay,Hritik and Utkarsha")
Write.configure(state="disabled", fg="black")
Write.pack(side="left")
Write.place(x=800, y=750)


window.mainloop()
