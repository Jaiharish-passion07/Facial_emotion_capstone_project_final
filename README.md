
# Face Emotion Recognition

A face emotion recognition system is used to identify emotions in humans like happiness, anger, sadness, etc. This approach is helpful in many real-world applications today, such as the education system. When using video conferencing in e-learning systems, this technology can identify the emotional states of the learners. This tech instantaneously communicates the students' emotional states to the teacher to promote a more engaging learning environment.

![cover_1](https://user-images.githubusercontent.com/78978975/182883197-b2ce4897-61f3-4966-92a5-01b7c197f088.jpg)

![cover_2](https://user-images.githubusercontent.com/78978975/182883228-eb091154-ce7c-4fd8-983e-6f708d69e01d.jpg)

## Aim of the Project

The main objective is to develop a system for recognising facial expressions of emotion utilising deep learning algorithms and convolution neural networks. It is a programme for recognising facial emotions.
## Table of Contents


| Contents        |
| ------------- |
| Dataset Summary      |
| Model Building    | 
| Real Time Application(Runs Locally) | 


## Dataset Summary

The face emotion dataset is fetched from Kaggle. The below Images show the summary of the dataset.

![collage](https://user-images.githubusercontent.com/78978975/182879453-04fc7bef-f6e9-4b8f-a1bd-efc5f37e63e3.jpg)


**Various Emotion Images**

![emo](https://user-images.githubusercontent.com/78978975/182881369-f5815c51-e021-4be4-b8da-f47afe6fea28.jpg)

[**Dataset_here**](https://www.kaggle.com/datasets/deadskull7/fer2013)

## Model Building


| Models        |
| ------------- |
| Mobilenet   |
|  Dexpression   | 
|  CNN| 
|Densenet|
|Resnet|

**Accuracy Plot**

![Screenshot (440)](https://user-images.githubusercontent.com/78978975/182879973-b9718377-bb94-47f7-820e-cc5636fcea18.png)

The ResNet model was chosen because it had the highest training accuracy of all
the models, and its validation accuracy was nearly 72 percent, which is
comparable to CNN models


[To view the Resnet Model binary file](https://github.com/Jaiharish-passion07/Facial_emotion_capstone_project_final/tree/master/model)

## Runs Locally

This project is developed as a Real time Applications by capturing live video from Local Machine 
and predict the each frame of emotions. To demonstrate this Applications lets discuss

**Installing the dependencies packages**

Requirements file can be accesssed [here](https://github.com/Jaiharish-passion07/Facial_emotion_capstone_project_final/blob/master/requirements.txt)


```bash
pip install -r requirements.txt

```
**Webapp Development using Streamlit Frameworks**


```bash
import streamlit as st
from PIL import Image
from model_predict import *
from streamlit_webrtc import webrtc_streamer

st.title("✨ Welcome ✨")
st.sidebar.title("🎇Choose an options🎇")
choice_options=st.sidebar.selectbox("",('Home','Start webcam','About'))

if choice_options=="Home":
    st.title('👨Face Emotion Recognition using Live Web Camera👩')
    image = Image.open('data/face.jpeg')
    st.image(image)
    st.sidebar.subheader("""💎 Face Emotion Recognition is a system used to detect the emotions from face.""")
    st.sidebar.subheader("""💎 Nowadays it is widely used applications.Eg: In zoom meeting we can able to detect the student emotion.""")
    st.sidebar.subheader("""💎 It is very helpful for teachers where they can able to teach based on their students emotion and make class more interactive.""")
if choice_options=="Start webcam":
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your face emotion")
    webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
if choice_options=="About":
    st.title('Project Members')
    col1, col2= st.columns(2)
    with col1:
        image_1= Image.open('data/ape.png')
        st.subheader("Ashik Kumar")
        st.image(image_1)
        st.write("Email:ashikkumar491@gmail.com")
        st.markdown("""[LinkedIn profile](https://www.linkedin.com/in/ashik-kumar-94a06a207)""")
    with col2:
        image_2 = Image.open('data/jai.png')
        st.subheader("Jai Harish S")
        st.image(image_2)
        st.write("Email:jaiharishs361@gmail.com")
        st.markdown("""[LinkedIn profile](https://www.linkedin.com/in/jai-harish-s-64b1b01ab)""")

    col1, col2= st.columns(2)
    with col1:
        image_3 = Image.open('data/pranil.png')
        st.subheader("Pranil Satish Thorat")
        st.image(image_3)
        st.write("Email:pranilthorat@gmail.com")
        st.markdown("""[LinkedIn profile](https://www.linkedin.com/in/pranil-thorat-834361216)""")
    with col2:
        image_4 = Image.open('data/saransh.png')
        st.subheader("Saransh Srivastava")
        st.image(image_4)
        st.write("Email:saranshoffice@gmail.com")
        st.markdown("""[LinkedIn profile](https://www.linkedin.com/in/saranshsrivastava13)""")
    col1, col2= st.columns(2)
    with col1:
        image_5 = Image.open('data/akasl.png')
        st.subheader("Bhaskar subanji")
        st.image(image_5)
        st.write("Email:bysubanji@gmail.com")
        st.markdown("""[LinkedIn profile](https://www.linkedin.com/in/bysubanji)""")
    with col2:
        pass

```
To access the ***app.py*** file you can click [here](https://github.com/Jaiharish-passion07/Facial_emotion_capstone_project_final/blob/master/app.py)

**Image preprocessing and prediction script**

```bash
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from streamlit_webrtc import VideoTransformerBase

# load model
emotion_dict = {0:'angry', 1 :'disgust', 2: 'fear', 3:'happy', 4: 'sad',5:'suprise',6:'neutral'}
classifier = load_model('model/final_model.h5')

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
except Exception:
    pass

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                label_position = (x, y)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        return img

```

To access the ***model_predict.py*** file you can click [here](https://github.com/Jaiharish-passion07/Facial_emotion_capstone_project_final/blob/master/model_predict.py)


## Model Building Notebook

To observe the model's construction, testing, and training. You may view a comparison of each model's accuracy and confusion matrix on the notebook.
While the Notebook includes all of the information

[Model_Building_Notebook](https://github.com/Jaiharish-passion07/Facial_emotion_capstone_project_final/blob/master/Model_Notebooks/Team_Final_Face_Emotion_Recognition_Notebook.ipynb)


## Tech Stack

- Python
- Streamlit
- Keras
- Tensorflow
- Image Preprocessing
- Convloutional Neural Network
- Transfer Learning Techniques
- Data Visualization

## Demonstration of Short Video

Demonstration of facial emotion recognition in real-time live video capture and recording video

https://user-images.githubusercontent.com/78978975/182880068-907a2f69-47f7-4020-b046-62a097a76a78.mp4
