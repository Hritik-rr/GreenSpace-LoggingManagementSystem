# GreenSpace: Logging & Management System
``` Not Final, Draft ```


<p align = "center">
<img src = "https://user-images.githubusercontent.com/41600508/126094519-e7a42ba0-166c-42e9-b642-860132379875.png" width = "350" height = "350" >
</p>


Green Space is an python based project which inherently uses OpenCV which is the spine of this project, the concept behind our project is to provide a work frame for 
defining the real time usage of a computer vision :camera: by the means of building a system that can be used for the verification as well as identification.
If any person whose record is already present in the system then it will show the record otherwise it will capture the image of person and save it as unidentified which
can be further labelled.

##  Table of Contents:

* [Purpose of the Project] (https://github.com/Hritik-rr/Logging-Management-System#table-of-contents) ```and http://www.google.fr/```
* [Purpose of the Project] (docs/Purpose.md)
    * [Abstract] (http://www.google.fr/ "Named link title")
* Solution with the help of application
* User level examples
```(Real time usage examples)```
* Different Modules
* Flow Diagram
* System Architecture
* [Installation Guide] (#installation)

## Purpose:
In our given set of project, we proposed to make an automated Staff and Visitor logging Management system. Our system consists of a camera setup for calibrating video faces, its further abilities include pre-processing of images and extracting facial features for face recognition. The different methods within the sequence of flow of application in our project consist of:
* Database Creation,
* Face Detection,
* Data Gathering and
* Face Recognition
   ### **Abstract:**
   This system will provide a format for propagating operation of real time usage of computer vision. The actual usage of our project is that when a person enters, then its entry time is recorded and stored in the csv file. Now, if any visitor enters (whose information/ data is not available in our database) in the block then its entity is registered and stored in the database and the stored info is used for future use. Our system can be used for verification as well as identification. If any person whose record is already present in the system, then it will show the record otherwise it will capture the image of the person and save it as unidentified which can be used further as per requirement.


## Modules:
***:one:.  Capturing and Storing Images:***

Asking the user to enter its name and then creating the folder having the entered name which will be used for labelling the Data. With the help of inbuilt Video capture method in openCV starting the process of capturing images of an individual, prompting the user to press the shutter in order to capture a minimum of 12 images(here, the capture module will capture and store 50 odd images of the subject) for better feature extraction. All the captured images are stored into the labeled folder with numerical name of the image(1.jpg,2.jpg,3.jpg etc).
Hence, Creating the database for further training.


***:two:.  Training the recognizer:***

We are using the LBPHFaceRecognizer with is inbuilt in the Cv2 library, we are using this Recognizer because it is not significantly affected by light and, in real life, we can't guarantee perfect light conditions, also we are using the Haar cascade Classifier for detecting the Region of Interest(Face).With the help of OS library walking into the Directory of our Database and reading images label by label, by using pillow opening image in Grayscale and converting into numpy Array and then passing this array into our Cascade Classifier to detect the face part of the image and then creating a list of obtained Region of Interest. Using the dump method of Pickle library serializing the label list object. In the end training our recognizer with the ROIs list and corresponding labels and finally saving the obtained YML file(trainer.yml)



<p align = "center">
<img src = "https://user-images.githubusercontent.com/41600508/126077345-7cef9f03-31fb-42ab-9551-3b5cf92cfed2.png" width = "880" height = "320" span style="display:block;text-align:center">
<img src = "https://user-images.githubusercontent.com/41600508/126077353-34e36f1d-3665-4691-aecd-6720076336f3.png" width = "880" height = "320" span style="display:block;text-align:center">
</p>


***:three:.  Prediction:***

This is where we get to see if our algorithm is recognizing our individual faces or not. We are implementing real time face detection and Recognition, using the same cascade classifier and the same recognizer that we used in our Training module. First loading the dumped object using pickle and de-serializing the list object for use in our script. Detecting the face(Region of Interest) of the image frame obtained from the device camera and passing this ROI as a parameter in our recognizers predict function, here we are using a confidence value that shows the lower bound of match percentage. Printing the ID corresponding to the given face.

         
**Face Detection**         |  **Face Recognition**
:-------------------------:|:-------------------------:
<img src = "https://user-images.githubusercontent.com/41600508/126092369-72ce9a0e-70c2-4dfd-b69c-203894c3cd17.png" width = "400" height = "420">  |  <img src = "https://user-images.githubusercontent.com/41600508/126092378-f4e4312a-bd90-46ee-97a9-eddbab9c2f97.png" width = "400" height = "420">


<br/>

## Flow Diagram:
<p align = "center">
<img src = "https://user-images.githubusercontent.com/41600508/126078018-cff95766-5559-4db2-ba6b-b173a74d7976.png" width = "690" height = "435" span style="display:block;text-align:center">
</p>


## Architecture Diagram:
<p align = "center">
<img src = "https://user-images.githubusercontent.com/41600508/126078234-9987a119-b57e-4727-83bb-b7b170e8b40e.png" width = "690" height = "420" span style="display:block;text-align:center">
</p>

## Installation
Requires python version 3.5 or later.
- python3 -m venv venv . Then activate your virtual environment(OS specific). (If you don't have venv, install it) [optional]
``` python -m pip install -r requirements.txt ```
``` python project.py ```