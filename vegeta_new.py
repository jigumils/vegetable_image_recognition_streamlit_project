import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


#Load model
model = tf.keras.models.load_model('vegetable_model_new.h5')

#label dictionary
dict_ = {
        0: 'Bean',
        1: 'Bitter_Gourd',
        2: 'Bottle_Gourd',
        3: 'Brinjal',
        4: 'Broccoli',
        5: 'Cabbage',
        6: 'Capsicum',
        7: 'Carrot',
        8: 'Cauliflower',
        9: 'Cucumber',
        10: 'Papaya',
        11: 'Potato',
        12: 'Pumpkin',
        13: 'Radish',
        14: 'Tomato'
        }


#Dataset gambar
paths_img = []
for i in range(15):
    paths_img.append('./data_pred/{}.jpg'.format(dict_[i]))


#Load dan show gambar
def image_show(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    return img


#predict gambar
def image_pred(img):
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    prediction = model.predict(img_preprocessed)
    pred = np.argmax(prediction)
    st.image(image_show(paths_img[pred]))
    return dict_[pred]


#Display
st.title("Vegetable Image Recognition App")
uploaded_file = st.file_uploader("Unggah gambar yang ingin direkognisi! (bisa pakai yang ada di folder data_pred)")
if uploaded_file != None:
    st.image(image_show(uploaded_file))

if st.button("Click Here to Classify"):
    img = image_show(uploaded_file)
    st.write(image_pred(img))