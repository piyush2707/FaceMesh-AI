
import io
import csv
import time

import av
import cv2
import numpy as np
import streamlit as st

from PIL import Image
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from face_engine import (
    create_landmarker,
    analyze_image,
    render_mesh,
    get_expression_data,
    get_landmark_data,
    process_video,
)

st.set_page_config(
    page_title="FaceMesh AI Studio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}
.hero {
    padding: 28px;
    border: 1px solid #193b55;
    border-radius: 18px;
    background: linear-gradient(120deg,#101c32,#092b3a);
    margin-bottom: 25px;
}
.hero h1 {
    color: #00e5ff;
    font-size: 38px;
    margin-bottom: 5px;
}
.hero p {
    color: #a9c4d8;
}
.metric-card {
    padding: 18px;
    background: #111b2d;
    border: 1px solid #25364c;
    border-radius: 12px;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid #24364a;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# SESSION STATE
# -----------------------------

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_landmarks" not in st.session_state:
    st.session_state.last_landmarks = []


# -----------------------------
# HEADER
# -----------------------------

st.markdown("""
<div class="hero">
    <h1>◈ FaceMesh AI Studio</h1>
    <p>
        Professional facial landmark detection,
        3D mesh visualization and expression analysis.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("⚙️ Studio Settings")

face_count = st.sidebar.slider(
    "Maximum detected faces",
    min_value=1,
    max_value=5,
    value=2,
)

show_mesh = st.sidebar.checkbox(
    "Show triangular mesh", True
)

show_contours = st.sidebar.checkbox(
    "Show facial contours", True
)

show_points = st.sidebar.checkbox(
    "Show individual landmarks", False
)

st.sidebar.divider()

st.sidebar.caption("FaceMesh AI Studio")
st.sidebar.caption("Powered by MediaPipe + OpenCV")


# -----------------------------
# LIVE CAMERA PROCESSOR
# -----------------------------

class LiveFaceProcessor(VideoProcessorBase):

    def __init__(self):
        self.detector = create_landmarker(
            "VIDEO",
            face_count,
        )
        self.mesh = show_mesh
        self.contours = show_contours
        self.points = show_points
        self.frames = 0
        self.last_time = time.time()
        self.fps = 0.0

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = __import__(
            "mediapipe"
        ).Image(
            image_format=__import__(
                "mediapipe"
            ).ImageFormat.SRGB,
            data=rgb,
        )

        timestamp = time.monotonic_ns() // 1_000_000

        result = self.detector.detect_for_video(
            mp_image,
            timestamp,
        )

        output = render_mesh(
            image,
            result,
            self.mesh,
            self.contours,
            self.points,
        )

        self.frames += 1
        now = time.time()

        if now - self.last_time >= 1:
            self.fps = self.frames / (
                now - self.last_time
            )
            self.frames = 0
            self.last_time = now

        cv2.putText(
            output,
            f"FPS: {self.fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24",
        )

    def __del__(self):
        try:
            self.detector.close()
        except Exception:
            pass


# -----------------------------
# TABS
# -----------------------------

tab_live, tab_photo, tab_video, tab_analysis = st.tabs([
    "📹 Live Camera",
    "🖼️ Photo Studio",
    "🎬 Video Studio",
    "🧠 Face Analysis",
])


# -----------------------------
# LIVE CAMERA
# -----------------------------

with tab_live:

    st.subheader("Real-time Face Tracking")

    st.write(
        "Allow camera access to see your facial mesh "
        "in real time."
    )

    webrtc_streamer(
        key="facemesh-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveFaceProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st.info(
        "Camera access requires browser permission. "
        "Use HTTPS when accessing the deployed application."
    )


# -----------------------------
# PHOTO STUDIO
# -----------------------------

with tab_photo:

    st.subheader("Image Face Mesh Generator")

    uploaded_image = st.file_uploader(
        "Upload a face photograph",
        type=["jpg", "jpeg", "png", "webp"],
        key="photo_upload",
    )

    if uploaded_image:

        image = Image.open(
            io.BytesIO(uploaded_image.getvalue())
        ).convert("RGB")

        original = np.array(image)
        bgr = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR,
        )

        if st.button(
            "Generate Face Mesh",
            type="primary",
            key="generate_photo",
        ):

            with st.spinner("Analyzing face..."):

                result = analyze_image(
                    bgr,
                    face_count,
                )

                annotated = render_mesh(
                    bgr,
                    result,
                    show_mesh,
                    show_contours,
                    show_points,
                )

                st.session_state.last_result = result
                st.session_state.last_landmarks = (
                    get_landmark_data(result)
                )

                st.session_state.photo_output = (
                    cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB,
                    )
                )

        if "photo_output" in st.session_state:

            output = st.session_state.photo_output

            left, right = st.columns(2)

            with left:
                st.image(
                    original,
                    caption="Original Image",
                    use_container_width=True,
                )

            with right:
                st.image(
                    output,
                    caption="FaceMesh Output",
                    use_container_width=True,
                )

            st.success(
                f"Detected {len(st.session_state.last_result.face_landmarks)} face(s)"
            )

            encoded = cv2.imencode(
                ".png",
                cv2.cvtColor(
                    output,
                    cv2.COLOR_RGB2BGR,
                ),
            )[1]

            st.download_button(
                "⬇️ Download Mesh Image",
                data=encoded.tobytes(),
                file_name="facemesh_output.png",
                mime="image/png",
            )


# -----------------------------
# VIDEO STUDIO
# -----------------------------

with tab_video:

    st.subheader("Video Face Mesh Processing")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload",
    )

    if uploaded_video:

        st.video(uploaded_video)

        if st.button(
            "Process Video",
            type="primary",
            key="process_video",
        ):

            progress = st.progress(0)
            status = st.empty()

            try:
                status.info(
                    "Processing video. Please wait..."
                )

                with st.spinner("Generating face mesh..."):

                    video_result = process_video(
                        uploaded_video,
                        show_mesh,
                        show_contours,
                        show_points,
                        face_count,
                    )

                progress.progress(100)

                status.success(
                    "Video processing completed!"
                )

                st.metric(
                    "Processed Frames",
                    video_result["processed"],
                )

                st.metric(
                    "Processing Time",
                    f'{video_result["elapsed"]} seconds',
                )

                with open(
                    video_result["path"],
                    "rb",
                ) as file:
                    video_bytes = file.read()

                st.video(video_bytes)

                st.download_button(
                    "⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name="facemesh_output.mp4",
                    mime="video/mp4",
                )

            except Exception as error:
                st.error(str(error))


# -----------------------------
# FACE ANALYSIS
# -----------------------------

with tab_analysis:

    st.subheader("Facial Landmark Analytics")

    result = st.session_state.last_result

    if result is None:

        st.info(
            "Generate a face mesh in Photo Studio "
            "to view facial analysis."
        )

    else:

        face_total = len(result.face_landmarks)

        st.metric(
            "Detected Faces",
            face_total,
        )

        if face_total > 0:

            selected_face = st.selectbox(
                "Select face",
                range(face_total),
            )

            landmarks = get_landmark_data(
                result,
                selected_face,
            )

            expressions = get_expression_data(
                result,
                selected_face,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "3D Landmarks",
                    len(landmarks),
                )

                st.write("Landmark coordinates")

                st.dataframe(
                    landmarks,
                    use_container_width=True,
                    height=350,
                )

                csv_buffer = io.StringIO()

                if landmarks:
                    writer = csv.DictWriter(
                        csv_buffer,
                        fieldnames=landmarks[0].keys(),
                    )
                    writer.writeheader()
                    writer.writerows(landmarks)

                st.download_button(
                    "⬇️ Export Landmarks CSV",
                    data=csv_buffer.getvalue(),
                    file_name="face_landmarks.csv",
                    mime="text/csv",
                )

            with col2:

                st.write("Facial Expression Blendshapes")

                if expressions:

                    sorted_expressions = sorted(
                        expressions.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )

                    for name, score in sorted_expressions[:15]:

                        st.write(
                            f"**{name}** — {score:.3f}"
                        )

                        st.progress(
                            min(max(score, 0.0), 1.0)
                        )

                else:
                    st.info(
                        "Expression data unavailable."
                    )


# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "FaceMesh AI Studio | Python Computer Vision | "
    "MediaPipe Face Landmarker"
)
