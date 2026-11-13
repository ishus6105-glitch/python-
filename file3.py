# Import Libraries
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the Model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Create User Interface
st.title("Iris Flower Classification")
st.write("Enter the flower measurements:")

# Add Input Widgets

# Text Inputs
sepal_length = st.text_input("Sepal Length (cm)", "5.1")
sepal_width = st.text_input("Sepal Width (cm)", "3.5")

# Selection Widgets (Dropdown / Slider)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.selectbox(
    "Petal Width (cm)",
    [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
)

# Prepare Input Data
input_data = pd.DataFrame({
    "sepal length (cm)": [float(sepal_length)],
    "sepal width (cm)": [float(sepal_width)],
    "petal length (cm)": [petal_length],
    "petal width (cm)": [petal_width]
})

# Make Prediction
prediction = model.predict(input_data)
species = iris.target_names[prediction[0]]

# Display Output
st.subheader("Prediction Result")
st.success(f"The predicted flower species is {species}")


# Import Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load Dataset
data = pd.read_csv("Bengaluru_House_Data.csv")

# Data Preprocessing

# Drop missing values
data = data.dropna()

# Convert total_sqft to numeric
def convert_sqft(x):
    try:
        if '-' in str(x):
            nums = x.split('-')
            return (float(nums[0]) + float(nums[1])) / 2
        return float(x)
    except:
        return None

data['total_sqft'] = data['total_sqft'].apply(convert_sqft)
data = data.dropna()

# Extract BHK from size column
data['bhk'] = data['size'].str.extract(r'(\d+)').astype(int)

# Select relevant features
data = data[['total_sqft', 'bath', 'bhk', 'price']]
data.columns = ['Area', 'Bathrooms', 'Bedrooms', 'Price']

# Prepare Data
X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Train Regression Model
model = LinearRegression()
model.fit(X, y)

# Create User Interface
st.title("Bengaluru Housing Price Prediction")
st.write("Enter property details to estimate price.")

# Input Widgets
area = st.number_input("Area (sq. ft)", 500, 10000, 1000)

bedrooms = st.selectbox(
    "Bedrooms (BHK)",
    [1, 2, 3, 4, 5]
)

bathrooms = st.selectbox(
    "Bathrooms",
    [1, 2, 3, 4]
)

# Prediction
input_data = pd.DataFrame(
    [[area, bedrooms, bathrooms]],
    columns=['Area', 'Bedrooms', 'Bathrooms']
)

prediction = model.predict(input_data)[0]

# Display Result
st.subheader("Predicted Price")
st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")

# Visualization
st.subheader("Prediction Visualization")

fig, ax = plt.subplots()
ax.bar(["Predicted Price"], [prediction])
ax.set_ylabel("Price (Lakhs)")
st.pyplot(fig)


# Import Libraries
import pandas as pd
import streamlit as st
import plotly.express as px

# Load Dataset
data = pd.read_csv("covid_19_india.csv")
st.write("Dataset Preview", data.head())

# Data Preprocessing

# Convert Date column
data['Date'] = pd.to_datetime(
    data['Date'],
    format='%Y/%m/%d',
    errors='coerce'
)

# Rename columns for easy use
data = data.rename(columns={
    'State/UT': 'State',
    'State/UnionTerritory': 'State',
    'Cured': 'Recovered'
})

# Drop missing values
data = data.dropna(subset=['Date', 'State'])

# Calculate Active cases if not present
if 'Active' not in data.columns:
    data['Active'] = (
        data['Confirmed'] -
        data['Recovered'] -
        data['Deaths']
    )

# Create Interactive UI
st.title("COVID-19 India Dashboard")

# State Filter
states = sorted(data['State'].unique())

selected_state = st.selectbox(
    "Select State/UT",
    states
)

# Date Filter
start_date = st.date_input(
    "Start Date",
    data['Date'].min().date()
)

end_date = st.date_input(
    "End Date",
    data['Date'].max().date()
)

# Filter Data
filtered_data = data[
    (data['State'] == selected_state) &
    (data['Date'] >= pd.to_datetime(start_date)) &
    (data['Date'] <= pd.to_datetime(end_date))
].sort_values('Date')

# Check Data
if filtered_data.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# Line Chart (Trend Analysis)
fig = px.line(
    filtered_data,
    x='Date',
    y=['Confirmed', 'Active', 'Recovered', 'Deaths'],
    title=f"COVID-19 Trends in {selected_state}"
)

st.plotly_chart(fig)

# Bar Chart (Latest Status)
latest_data = filtered_data.iloc[-1]

fig_bar = px.bar(
    x=['Confirmed', 'Active', 'Recovered', 'Deaths'],
    y=[
        latest_data['Confirmed'],
        latest_data['Active'],
        latest_data['Recovered'],
        latest_data['Deaths']
    ],
    title="Latest COVID-19 Status"
)

st.plotly_chart(fig_bar)

# Display Key Metrics
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Confirmed",
        int(latest_data['Confirmed'])
    )

with col2:
    st.metric(
        "Active",
        int(latest_data['Active'])
    )

with col3:
    st.metric(
        "Recovered",
        int(latest_data['Recovered'])
    )

with col4:
    st.metric(
        "Deaths",
        int(latest_data['Deaths'])
    )

# Calculate Recovery and Mortality Rates
confirmed = latest_data['Confirmed']

if confirmed > 0:
    recovery_rate = (
        latest_data['Recovered'] / confirmed
    ) * 100

    mortality_rate = (
        latest_data['Deaths'] / confirmed
    ) * 100
else:
    recovery_rate = 0
    mortality_rate = 0

st.metric(
    "Recovery Rate (%)",
    round(recovery_rate, 2)
)

st.metric(
    "Mortality Rate (%)",
    round(mortality_rate, 2)
)



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