import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page content
st.title("Student Performance Dashboard")
st.write("This dashboard shows sample data about student marks.")

# Create sample data
# Marks of 5 students over 3 tests

data = {
    "Test 1": [65, 70, 80, 75, 90],
    "Test 2": [68, 72, 78, 80, 92],
    "Test 3": [70, 75, 82, 85, 95]
}

# Convert data into a DataFrame
df = pd.DataFrame(
    data,
    index=[
        "Student A",
        "Student B",
        "Student C",
        "Student D",
        "Student E"
    ]
)

# Display the DataFrame
st.subheader("Marks Data")
st.dataframe(df)

# Display a line chart
st.subheader("Performance Trend (Line Chart)")
st.line_chart(df.T)




import streamlit as st

st.title("Student Performance Dashboard")
st.success("Data loaded successfully.")
st.warning("Some records contain missing values.")
st.error("Failed to connect to the database.")
st.info("Analysis is in progress.")


import streamlit as st
import pandas as pd
from datetime import date, time

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Streamlit Widgets Demo",
    page_icon="🎯",
    layout="wide"
)

st.title("Streamlit Widgets Demonstration")
st.write("Learn different input, action, and display widgets in Streamlit.")

st.divider()


# ==========================================
# PART 1: TEXT AND INPUT WIDGETS
# ==========================================

st.header("1. Text and Input Widgets")

# 1. Text Input
st.subheader("1. st.text_input()")

name = st.text_input("Enter your name:")

st.write("Your Name:", name)


# 2. Text Area
st.subheader("2. st.text_area()")

description = st.text_area("Enter your description:")

st.write("Description:", description)


# 3. Number Input
st.subheader("3. st.number_input()")

age = st.number_input(
    "Enter your age:",
    min_value=1,
    max_value=100,
    value=25
)

st.write("Your Age:", age)


# 4. Date Input
st.subheader("4. st.date_input()")

selected_date = st.date_input(
    "Select your date:",
    value=date.today()
)

st.write("Selected Date:", selected_date)


# 5. Time Input
st.subheader("5. st.time_input()")

selected_time = st.time_input(
    "Select your time:",
    value=time(10, 30)
)

st.write("Selected Time:", selected_time)


st.divider()


# ==========================================
# PART 2: ACTION WIDGETS
# ==========================================

st.header("2. Action Widgets")


# 1. Button
st.subheader("1. st.button()")

if st.button("Click Me"):
    st.success("Button Clicked Successfully!")


# 2. Slider
st.subheader("2. st.slider()")

marks = st.slider(
    "Select your marks:",
    min_value=0,
    max_value=100,
    value=50
)

st.write("Your Marks:", marks)


# 3. Select Slider
st.subheader("3. st.select_slider()")

rating = st.select_slider(
    "Select your performance:",
    options=["Poor", "Average", "Good", "Excellent"]
)

st.write("Your Performance:", rating)


# 4. Selectbox
st.subheader("4. st.selectbox()")

course = st.selectbox(
    "Select your course:",
    ["Python", "Data Science", "Artificial Intelligence", "Machine Learning"]
)

st.write("Selected Course:", course)


# 5. Radio Button
st.subheader("5. st.radio()")

gender = st.radio(
    "Select your gender:",
    ["Male", "Female", "Other"]
)

st.write("Selected Option:", gender)


# 6. Checkbox
st.subheader("6. st.checkbox()")

show_data = st.checkbox("Show Student Information")

if show_data:
    st.info("Student Name: Rahul")
    st.info("Course: Python")
    st.info("Age: 22")


# 7. Multiselect
st.subheader("7. st.multiselect()")

skills = st.multiselect(
    "Select your skills:",
    ["Python", "SQL", "Machine Learning", "Deep Learning", "Power BI"]
)

st.write("Selected Skills:", skills)


st.divider()


# ==========================================
# PART 3: DATA DISPLAY WIDGETS
# ==========================================

st.header("3. Data Display Widgets")


# Sample Dataset
data = {
    "Name": ["Rahul", "Priya", "Amit", "Sneha", "Rohit"],
    "Age": [22, 23, 24, 21, 25],
    "Course": ["Python", "AI", "ML", "Data Science", "Python"],
    "Marks": [85, 90, 78, 95, 88]
}

df = pd.DataFrame(data)


# 1. Dataframe
st.subheader("1. st.dataframe()")

st.write("Interactive Dataset:")

st.dataframe(df, use_container_width=True)


# 2. Table
st.subheader("2. st.table()")

st.write("Static Dataset:")

st.table(df)


# 3. Metric
st.subheader("3. st.metric()")

col1, col2, col3 = st.columns(3)

col1.metric(
    label="Total Students",
    value=5
)

col2.metric(
    label="Average Marks",
    value="87.2"
)

col3.metric(
    label="Highest Marks",
    value="95"
)


# 4. JSON
st.subheader("4. st.json()")

student = {
    "Name": "Rahul",
    "Age": 22,
    "Course": "Artificial Intelligence",
    "Skills": ["Python", "SQL", "Machine Learning"],
    "Passed": True
}

st.json(student)


# 5. Write
st.subheader("5. st.write()")

st.write("Welcome to Streamlit!")

st.write("Student Dataset:", df)

st.write("Student Names:", df["Name"])

st.write("Average Marks:", df["Marks"].mean())


st.divider()

st.success("All Streamlit Widgets Demonstrated Successfully!")



import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Media Files Demo",
    page_icon="🎬",
    layout="wide"
)

# Application Title
st.title("🎬 Display Elements – Media Files")

st.write(
    "This application demonstrates how to display and interact "
    "with images, audio, and video files using Streamlit."
)

st.divider()

# ------------------------------------------------
# 1. Display Image using st.image()
# ------------------------------------------------

st.header("1. Display Image")

st.image(
    "https://picsum.photos/800/400",
    caption="Sample Image",
    use_container_width=True
)

st.divider()

# ------------------------------------------------
# 2. Display Multiple Images
# ------------------------------------------------

st.header("2. Display Multiple Images")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(
        "https://picsum.photos/id/1015/300/200",
        caption="Mountain"
    )

with col2:
    st.image(
        "https://picsum.photos/id/1016/300/200",
        caption="Forest"
    )

with col3:
    st.image(
        "https://picsum.photos/id/1018/300/200",
        caption="Nature"
    )

st.divider()

# ------------------------------------------------
# 3. Play Audio using st.audio()
# ------------------------------------------------

st.header("3. Play Audio")

audio_url = (
    "https://www.soundhelix.com/examples/mp3/"
    "SoundHelix-Song-1.mp3"
)

st.audio(audio_url, format="audio/mp3")

st.divider()

# ------------------------------------------------
# 4. Play Video using st.video()
# ------------------------------------------------

st.header("4. Play Video")

video_url = (
    "https://www.w3schools.com/html/"
    "mov_bbb.mp4"
)

st.video(video_url)

st.divider()

# ------------------------------------------------
# 5. Display Media from URLs
# ------------------------------------------------

st.header("5. Media from URLs")

st.subheader("Display Image from URL")

image_url = st.text_input(
    "Enter Image URL:",
    "https://picsum.photos/600/300"
)

if image_url:
    st.image(image_url, caption="Image from URL")

st.subheader("Play Audio from URL")

audio_input = st.text_input(
    "Enter Audio URL:",
    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
)

if audio_input:
    st.audio(audio_input)

st.subheader("Play Video from URL")

video_input = st.text_input(
    "Enter Video URL:",
    "https://www.w3schools.com/html/mov_bbb.mp4"
)

if video_input:
    st.video(video_input)

st.success("Media Files Demonstration Completed!")



import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page content
st.title("Student Performance Dashboard")
st.write("This dashboard shows sample data about student marks.")

# Create sample data
# Marks of 5 students over 3 tests

data = {
    "Test 1": [65, 70, 80, 75, 90],
    "Test 2": [68, 72, 78, 80, 92],
    "Test 3": [70, 75, 82, 85, 95]
}

# Convert data into a DataFrame
df = pd.DataFrame(
    data,
    index=[
        "Student A",
        "Student B",
        "Student C",
        "Student D",
        "Student E"
    ]
)

# Display the DataFrame
st.subheader("Marks Data")
st.dataframe(df)

# Display a line chart
st.subheader("Performance Trend (bar Chart)")
st.bar_chart(df.T)
