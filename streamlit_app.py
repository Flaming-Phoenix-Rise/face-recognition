import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import model_from_json
import os
from streamlit_chat import message
from pygame import mixer
from dotenv import load_dotenv
import google.generativeai as genai

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

labels = {
    0: 'angry. I want to help you',
    1: 'disgust. You can now handle it.',
    2: 'fear. Take a deep breath.',
    3: 'happy. Congrats on your nice day.',
    4: 'neutral...',
    5: 'sad. hope you feel better',
    6: 'stressed. please have some rest.',
    7: 'surprise. I know, is that not wonderful?'
}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def fun(user_role):
    if user_role == 'model':
        return 'assistant'
    else:
        return user_role

def emofeedback(prev):
    message(prev)
    if prev == 'happy. Congrats on your nice day.':
        message("Enjoy the Moment Fully")
        message("Sometimes, we rush through happy moments without truly enjoying them. Take a pause and practice mindfulness—observe the sensations, thoughts, and feelings without trying to intensify or suppress them.")
    elif prev == 'neutral...':
        message("Embrace the Calm")
        message("Neutral feelings can be a relief from more intense emotions. Try to view neutrality as a time of emotional rest, an opportunity to recharge.")
    elif prev == 'sad. hope you feel better':
        message("Here are some steps to deal with this emotion")
        message("1. Let yourself experience the sadness without rushing to push it away.")
        message("2. Sharing your feelings with a friend, family member, or therapist can lighten the emotional burden.")
    elif prev == 'angry. I want to help you':
        message("Here are some steps to deal with this emotion")
        message("1. Focus on slow, deep breaths to calm the physical tension.")
    elif prev == 'disgust. You can now handle it.':
        message("Here are some steps to deal with this emotion")
        message("1. Try to identify what specifically triggers your disgust.")
    elif prev == 'fear. Take a deep breath.':
        message("Here are some steps to deal with this emotion")
        message("1. Recognize the fear and name it.")
    elif prev == 'stressed. please have some rest.':
        message("Here are some steps to deal with this emotion")
        message("1. Mindfulness can help reduce stress.")
    elif prev == 'surprise. I know, is that not wonderful?':
        message("Recognize your emotional response. It’s okay to feel surprised.")

class VideoTransformer:
    def __init__(self):
        self.prev = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (p, q, r, s) in faces:
            face = gray[q:q + s, p:p + r]
            face_resized = cv2.resize(face, (48, 48))
            img_features = extract_features(face_resized)

            pred = model.predict(img_features)
            pred_class = np.argmax(pred)
            prediction_label = labels[pred_class]

            cv2.rectangle(img, (p, q), (p + r, q + s), (255, 0, 0), 2)
            cv2.putText(img, f'Emotion: {prediction_label}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if self.prev != prediction_label:
                self.prev = prediction_label
                emofeedback(prediction_label)
                if prediction_label == "neutral...":
                    st.audio("music.mp3", format="audio/mpeg", loop=True)
                elif prediction_label == "happy. Congrats on your nice day.":
                    st.audio("happy.mp3", format="audio/mpeg", loop=True)
                elif prediction_label == "sad. hope you feel better":
                    st.audio("sad.mp3", format="audio/mpeg", loop=True)
                elif prediction_label == "stressed. please have some rest.":
                    st.audio("stress.mp3", format="audio/mpeg", loop=True)

        return img

def chat():
    st.title("Chatbot to help you")
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key='AIzaSyAhIOrkXfbcD_LP0uhxoDIYbjy8MMTTarA')
    model = genai.GenerativeModel('gemini-pro')

    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    for message in st.session_state.chat_session.history:
        with st.chat_message(fun(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt = st.chat_input('Enter your Question')
    if user_prompt:
        st.chat_message('user').markdown(user_prompt)
        response = st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message('assistant'):
            st.markdown(response.text)

def main():
    st.title("Emotion Recognition and Chatbot")

    on = st.checkbox("Activate chatbot to get suggestions")

    camera_input = st.camera_input("Capture your face for emotion detection")
    if camera_input is not None:
        image = camera_input.getvalue()
        np_image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is not None:
            VideoTransformer().transform(image)

    if on:
        chat()

if __name__ == "__main__":
    main()
