# Import Libraries
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load and Train MNIST Model
@st.cache_resource
def train_model():

    # Load Dataset
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize Data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Build Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            input_shape=(28, 28, 1)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu'
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile Model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train Model
    model.fit(x_train, y_train, epochs=3)

    return model


# Load Model
model = train_model()

# Create User Interface
st.title("Digit Recognition App")
st.write("Draw a digit or upload an image to predict.")

# Drawing Canvas
st.subheader("Draw a Digit")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Upload Image Option
st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)


# Image Preprocessing
def preprocess_image(img):

    # Convert to Grayscale
    img = img.convert("L")

    # Convert to Array
    img = np.array(img)

    # Make Background Black
    if np.mean(img) > 127:
        img = 255 - img

    # Resize Image
    img = Image.fromarray(img)
    img = img.resize((28, 28))

    # Normalize Data
    img = np.array(img) / 255.0

    # Reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    return img


# Handle Input (Draw or Upload)
image = None

# From Canvas
if canvas_result.image_data is not None:
    if np.any(canvas_result.image_data[:, :, 3] > 0):
        image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

# From Upload
if uploaded_file is not None:
    image = Image.open(uploaded_file)

# Display Input Image
if image is not None:
    st.image(image, caption="Input Image", width=150)

    # Prediction
    processed = preprocess_image(image)
    prediction = model.predict(processed)

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Show Result
    st.subheader("Prediction")

    st.success(
        f"Predicted Digit: {predicted_digit}"
    )

    st.write(f"Confidence: {confidence:.2f}%")