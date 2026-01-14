from PIL import Image
import numpy as np
import streamlit as sl
import tensorflow as tf

@sl.cache_resource
def load_model():
 model = tf.keras.models.load_model("cats_dogs.keras")
 return model
def preprocess_image(image):
 image = image.convert("RGB")
 image = image.resize((150, 150))
 img_array = np.array(image)
 img_array = img_array / 255.0
 # Batch-Dimension hinzufügen (Modell erwartet Form (1, 150, 150, 3))
 img_array = np.expand_dims(img_array, axis=0)
 return img_array

model = load_model()

sl.title("Cat or Dog")
sl.write("Not so educated guesses on whether this is a Dog or a Cat")
uploaded_file = sl.file_uploader("Bild auswählen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
 image = Image.open(uploaded_file)
 # Bild anzeigen
 sl.image(image, caption="Hochgeladenes Bild", use_container_width=True)
 # Button Klassifizierung
 if sl.button("Klassifizieren"):
    sl.write("Verarbeite Bild...")

    # Vor dem Klassifizieren das Bild entsprechend vorverarbeiten
    x = preprocess_image(image)
    # Vorhersage
    pred = model.predict(x)[0][0]
    if pred < 0.5:
        label = "Cat"
        pred = 1 - pred
    else:
        label = "Dog"

    sl.write("Ergebnis:", label)
    sl.write(f"Wahrscheinlichkeit für {label}:", float(pred))

