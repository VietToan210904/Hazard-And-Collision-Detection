import streamlit as st
import base64
import subprocess
import os

st.set_page_config(page_title = "Hazard Detection", layout = "wide")

if "app_page" not in st.session_state:
    st.session_state.app_page = "intro"

def go_to(page_name):
    st.session_state.app_page = page_name

def set_bg_from_local(img_path):
    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    html, body, .stApp {{
        height: 100%;
        margin: 0;
    }}
    .stApp {{
        background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
        background-size: cover;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    }}
    </style>
    """, unsafe_allow_html = True)

def show_intro():
    set_bg_from_local("C:/Users/tonyh_yxuq8za/Desktop/HACD/Inference_PostProcessing/background.png")

    st.markdown("""
    <style>
    .stApp {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        overflow: hidden;
    }

    @keyframes fadeSlideUp {
        0% {
            opacity: 0;
            transform: translateY(40px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .intro-box {
        text-align: center;
        animation: fadeSlideUp 1.2s ease-out;
    }

    div.stButton > button {
        font-size: 1.1rem;
        font-weight: 500;
        padding: 0.7rem 2.2rem;
        border-radius: 12px;
        background-color: rgba(0, 150, 136, 0.9);
        color: white;
        border: none;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        cursor: pointer;
        transition: transform 0.3s ease, background-color 0.3s ease;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
        background-color: rgba(0, 200, 160, 1);
    }
    </style>
    """, unsafe_allow_html = True)

    st.markdown('<div class="intro-box">', unsafe_allow_html = True)

    st.markdown("""
    <h1 style="
        color: white;
        font-size: clamp(2rem, 5vw, 4rem);
        font-weight: 600;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 3px black;
        font-family: 'Poppins', sans-serif;
        animation: fadeSlideUp 1.2s ease-out;
    ">
    Hazard and Collision Detection in Real-Time
    </h1>
    """, unsafe_allow_html = True)

    if st.button("Let's Get Started"):
        st.session_state.app_page = "inference"

    st.markdown('</div>', unsafe_allow_html = True)

def show_inference_launcher():
    st.title("🚗 Hazard and Collision Detection in Real-Time")

    input_mode = st.radio("Choose input source:", ["📱 DroidCam (iPhone)", "📁 Upload Video File"])

    video_path = None
    if input_mode == "📁 Upload Video File":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file:
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

    if "infer_process" not in st.session_state:
        st.session_state.infer_process = None

    if st.button("▶ Run Inference"):
        if st.session_state.infer_process:
            st.warning("Inference is already running.")
        else:
            if input_mode == "📱 DroidCam (iPhone)":
                proc = subprocess.Popen(["python", "inference_app.py", "--source", "droidcam"])
                st.session_state.infer_process = proc
                st.success("Running inference on DroidCam.")
            elif input_mode == "📁 Upload Video File" and video_path:
                proc = subprocess.Popen(["python", "inference_app.py", "--source", video_path])
                st.session_state.infer_process = proc
                st.success("Running inference on uploaded video.")
            else:
                st.warning("Please upload a video first.")

    if st.session_state.infer_process and st.button("🛑 Stop Inference"):
        st.session_state.infer_process.terminate()
        st.session_state.infer_process = None
        st.success("Inference stopped.")


if st.session_state.app_page == "intro":
    show_intro()
elif st.session_state.app_page == "inference":
    show_inference_launcher()
