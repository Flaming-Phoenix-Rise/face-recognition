import streamlit as st
import cv2
import numpy as np
from huggingface_hub import metadata_save
from tensorflow.keras.models import model_from_json
from pygame import mixer
from streamlit_chat import message
from dotenv import load_dotenv
import streamlit as st
import os
import transformers
import google.generativeai as genai

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


labels = {
    0: 'angry. I want to help you',
    1: 'disgust. You can now handle it.',
    2: 'fear. Take a deep breath.',
    3: 'happy. Congrats on your nice day.',
    4: 'neutral...',
    5: 'sad. hope you feel better',
    6: 'stressed. pls have some rest.',
    7: 'surprise. I know, is that not wonderful?'
}
def fun(user_role):
    if user_role == 'model':
        return 'assistant'
    else:
        return user_role

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


def emofeedback(prev):
    if prev == 'happy. Congrats on your nice day.':
        message("Enjoy the Moment Fully")
        message("Sometimes, we rush through happy moments without truly enjoying them. Take a pause and practice mindfulness—observe the sensations, thoughts, and feelings without trying to intensify or suppress them. Just being present can deepen your experience and make the joy more manageable.")
    if prev == 'neutral...':
        message("Embrace the Calm")
        message("Neutral feelings can be a relief from more intense emotions. Try to view neutrality as a time of emotional rest, an opportunity to recharge, and a natural part of the emotional spectrum.")
        message("Use this time to observe your thoughts, surroundings, and internal experiences without judgment. Practicing mindfulness can help you develop a stronger awareness of subtle emotions, leading to a deeper understanding of yourself even when things seem uneventful.")
    if prev == 'sad. hope you feel better':
        message("Here are some steps to deal with this emotion")
        message("1. Let yourself experience the sadness without rushing to push it away. Avoiding or bottling up feelings can sometimes prolong them. Giving yourself permission to feel sad can be a crucial step toward processing it.")
        message("2. Sharing your feelings with a friend, family member, or therapist can lighten the emotional burden. Sometimes, just being heard and validated can provide relief and perspective.")
        message("3. Put together a box or collection of items that help you feel better—like comforting snacks, a journal, favorite quotes, or a soft blanket. Having these ready can be a helpful reminder of things that bring you peace.")
    if prev == 'angry. I want to help you':
        message("Here are some steps to deal with this emotion")
        message("1. Taking a moment to focus on slow, deep breaths can calm the physical tension that often comes with anger. Techniques like inhaling deeply for a count of four, holding for four, and exhaling for four can help you feel more in control.")
        message("2. Try to explore what triggered your anger. Ask yourself if it’s due to a specific situation, feeling of injustice, or past frustration. Understanding the reason behind your anger can provide clarity on how to handle it.")
        message("3. Step away from the situation if possible. A short break allows your emotions to settle and gives you a chance to come back with a cooler head.")
    if prev == 'disgust. You can now handle it.':
        message("Here are some steps to deal with this emotion")
        message("1. Try to identify what specifically triggers your disgust. Disgust can stem from a range of things—physical sensations, smells, visuals, or even ideas or behaviors. Understanding the specific trigger can help reduce the intensity of the reaction.")
        message("2. Deep breathing, mindfulness, or meditation can help ease physical reactions to disgust. When you feel disgust, focus on grounding exercises, like breathing deeply or counting, to regain a sense of calm.")
        message("3. Disgust is often a protective emotion that signals when something may be harmful or unhealthy. View it as a useful indicator rather than a burden, which can help you respond thoughtfully rather than reactively.")
    if prev == 'fear. Take a deep breath.':
        message("Here are some steps to deal with this emotion")
        message("1. Instead of pushing it away, recognize the fear and name it. Say to yourself, “I am feeling afraid because…” Being honest with yourself can make the emotion feel more manageable and less overwhelming.")
        message("2. Slow, deep breaths activate the parasympathetic nervous system, which helps calm the body’s stress response. Try breathing in for a count of four, holding for four, and exhaling for six. Repeat this until you feel a sense of calm.")
        message("3. If you can, break down what you fear into small, manageable steps. Gradually exposing yourself to what scares you can build resilience over time. Start with the least frightening aspect and work your way up as you feel more comfortable.")
    if prev == 'stressed. pls have some rest.':
        message("Here are some steps to deal with this emotion")
        message("1. Mindfulness involves staying present and aware of your thoughts and feelings without judgment. Mindfulness meditation can help reduce stress by promoting relaxation and acceptance.")
        message("2. Make time for activities you enjoy, whether it’s reading, gardening, painting, or playing a musical instrument. Engaging in hobbies can help distract you from stress and boost your mood.")
    if prev == 'surprise. I know, is that not wonderful?':
        message("Recognize your emotional response. It’s okay to feel surprised, whether it’s joy, confusion, or anxiety. Accepting your feelings can help you move through them.")
    message("To get other suggestions, chat with the following bot")
    # chat()

mixer.init()
def main():
    st.title("Facial Emotional Recognition")
    video_placeholder = st.empty()
    webcam = cv2.VideoCapture(0)
    prev = ""
    on = st.toggle("Activate chatbot to get suggestions")
    mu = st.toggle("Would you like to stop the music?")
    st.session_state.run = True
    while st.session_state.run:
        ret, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (p, q, r, s) in faces:
            face = gray[q:q + s, p:p + r]
            face_resized = cv2.resize(face, (48, 48))
            img = extract_features(face_resized)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
            cv2.putText(frame, f'Emotion: {prediction_label}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if prediction_label != prev:
                prev = prediction_label
                mixer.stop()
                if prev == "neutral...":
                    mixer.music.load("music.mp3")
                    mixer.music.set_volume(0.7)
                    mixer.music.play()
                elif prev == "happy. Congrats on your nice day.":
                    mixer.music.load("happy.mp3")
                    mixer.music.set_volume(0.7)
                    mixer.music.play()
                elif prev == "sad. hope you feel better":
                    mixer.music.load("sad.mp3")
                    mixer.music.set_volume(0.7)
                    mixer.music.play()

        video_placeholder.image(frame, channels="BGR")
        if mu:
            if mixer.music.get_busy():
                mixer.music.stop()
        if on:
            st.session_state.run = False
    message("It seems you are feeling")
    message(prev)
    message("Hello bot!", is_user=True)
    message("Can you give me some suggestions?", is_user=True)
    emofeedback(prev)
    if mu:
        mixer.music.stop()

if __name__ == "__main__":
    main()
