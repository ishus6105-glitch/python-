# ============================================================
# LAB 4: DIGIT RECOGNITION INTERFACE USING MNIST DATASET
# Technologies: Python, TensorFlow, CNN, Streamlit, NumPy, PIL
# ============================================================

# STEP 1: IMPORT REQUIRED LIBRARIES

import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# STEP 2: STREAMLIT PAGE CONFIGURATION

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")
st.write(
    "Draw a handwritten digit or upload an image. "
    "The CNN model will predict the digit from 0 to 9."
)

st.divider()


# STEP 3: LOAD MNIST DATASET AND TRAIN CNN MODEL

@st.cache_resource(show_spinner="Training the CNN model on MNIST...")

def train_model():

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize pixel values between 0 and 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape images for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build CNN model
    model = tf.keras.models.Sequential([

        tf.keras.layers.Input(shape=(28, 28, 1)),

        # First convolutional layer
        tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu"
        ),

        # First pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer
        tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        # Second pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convert feature maps into one-dimensional data
        tf.keras.layers.Flatten(),

        # Fully connected layer
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        # Prevent overfitting
        tf.keras.layers.Dropout(0.3),

        # Output layer for digits 0 to 9
        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate model on test dataset
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    return model, test_accuracy


# Train model or reuse the cached model
model, test_accuracy = train_model()

st.success("CNN model is ready!")

st.metric(
    "MNIST Test Accuracy",
    f"{test_accuracy * 100:.2f}%"
)

st.divider()


# STEP 4: CREATE DRAWING CANVAS

st.subheader("Option 1: Draw a Digit")

st.write("Use your mouse to draw a digit between 0 and 9.")

canvas_result = st_canvas(

    fill_color="rgba(0, 0, 0, 0)",

    stroke_width=18,

    stroke_color="#FFFFFF",

    background_color="#000000",

    height=280,

    width=280,

    drawing_mode="freedraw",

    key="digit_canvas"
)


# STEP 5: IMAGE UPLOAD OPTION

st.subheader("Option 2: Upload an Image")

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)


# STEP 6: IMAGE PREPROCESSING FUNCTION

def preprocess_image(image):

    # Convert image to grayscale
    gray = np.array(
        image.convert("L"),
        dtype=np.uint8
    )

    # Check image background using its border
    border_pixels = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1]
    ])

    # MNIST expects a white digit on a black background.
    # Invert images with a predominantly white background.
    if np.median(border_pixels) > 127:
        gray = 255 - gray

    # Identify the digit and remove the surrounding background
    mask = gray > 25

    coordinates = np.argwhere(mask)

    # Return None if no digit is found
    if coordinates.size == 0:
        return None, None

    # Find the digit boundaries
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)

    # Crop the digit
    digit = gray[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Get digit dimensions
    height, width = digit.shape

    # Resize while maintaining the original aspect ratio
    scale = 20.0 / max(height, width)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    # Resize using high-quality interpolation
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    digit_image = Image.fromarray(digit).resize(
        (new_width, new_height),
        resample=resampling
    )

    digit_array = np.array(digit_image, dtype=np.uint8)

    # Create a blank 28x28 MNIST-compatible image
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Place the resized digit in the center
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = digit_array

    # Center the digit using its center of mass
    total_mass = float(canvas.sum())

    if total_mass > 0:

        y_grid, x_grid = np.indices(canvas.shape)

        center_y = float(
            (y_grid * canvas).sum() / total_mass
        )

        center_x = float(
            (x_grid * canvas).sum() / total_mass
        )

        shift_y = int(round(13.5 - center_y))
        shift_x = int(round(13.5 - center_x))

        shifted = np.zeros_like(canvas)

        # Calculate source and destination coordinates
        source_y_start = max(0, -shift_y)
        source_y_end = min(28, 28 - shift_y)

        destination_y_start = max(0, shift_y)
        destination_y_end = (
            destination_y_start
            + source_y_end
            - source_y_start
        )

        source_x_start = max(0, -shift_x)
        source_x_end = min(28, 28 - shift_x)

        destination_x_start = max(0, shift_x)
        destination_x_end = (
            destination_x_start
            + source_x_end
            - source_x_start
        )

        # Shift the digit without losing image dimensions
        shifted[
            destination_y_start:destination_y_end,
            destination_x_start:destination_x_end
        ] = canvas[
            source_y_start:source_y_end,
            source_x_start:source_x_end
        ]

        canvas = shifted

    # Normalize pixel values between 0 and 1
    normalized = canvas.astype("float32") / 255.0

    # Reshape for CNN input: (1, 28, 28, 1)
    processed = normalized.reshape(1, 28, 28, 1)

    return processed, Image.fromarray(canvas)


# STEP 7: HANDLE DRAWING OR UPLOADED IMAGE

input_image = None

# Uploaded image gets priority when provided
if uploaded_file is not None:

    try:
        with Image.open(uploaded_file) as uploaded_image:
            input_image = uploaded_image.convert("RGB")

    except Exception as error:
        st.error(f"Unable to open the uploaded image: {error}")

# Otherwise, use the drawing canvas
elif canvas_result.image_data is not None:

    canvas_array = canvas_result.image_data.astype("uint8")

    # Check whether the user has actually drawn something
    if np.max(canvas_array[:, :, :3]) > 20:

        input_image = Image.fromarray(
            canvas_array,
            mode="RGBA"
        ).convert("RGB")


# STEP 8: DISPLAY INPUT IMAGE AND MAKE PREDICTION

if input_image is not None:

    st.divider()

    st.subheader("Input Image")

    st.image(
        input_image,
        caption="Selected handwritten digit",
        width=200
    )

    # Preprocess image
    processed_image, preview_image = preprocess_image(
        input_image
    )

    if processed_image is None:

        st.warning(
            "No handwritten digit detected. "
            "Please draw or upload a clearer digit."
        )

    else:

        # Display processed 28x28 image
        st.subheader("Preprocessed Image")

        st.image(
            preview_image,
            caption="28 × 28 MNIST model input",
            width=150
        )

        # Make prediction
        prediction = model.predict(
            processed_image,
            verbose=0
        )

        # Find predicted digit
        predicted_digit = int(
            np.argmax(prediction[0])
        )

        # Calculate prediction confidence
        confidence = float(
            np.max(prediction[0]) * 100
        )

        # Display result
        st.divider()

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # Display probability for each digit
        st.subheader("Prediction Probabilities")

        probability_data = {
            "Digit": list(range(10)),
            "Probability (%)": np.round(
                prediction[0] * 100,
                2
            )
        }

        st.dataframe(
            probability_data,
            hide_index=True,
            use_container_width=True
        )

else:

    st.info(
        "Draw a digit on the canvas or upload an image "
        "to get a prediction."
    )


# STEP 9: FOOTER

st.divider()

st.caption(
    "Developed using Python, TensorFlow, CNN, "
    "MNIST Dataset and Streamlit."
)