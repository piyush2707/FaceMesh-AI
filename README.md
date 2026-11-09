
# FaceMesh AI Studio

Professional facial landmark detection and 3D mesh visualization using Python.

## Features

- Live webcam face tracking
- Photo-based face mesh generation
- Video face mesh processing
- 478 3D facial landmarks
- Facial contours and triangular tessellation
- 52 facial expression blendshapes
- Multi-face detection
- Landmark CSV export
- Annotated PNG and MP4 export
- Adjustable visualization settings

## Technology

- Python 3.11
- MediaPipe Tasks
- OpenCV
- Streamlit
- streamlit-webrtc
- NumPy

## Project Structure

```text
FaceMesh-AI/
├── app.py
├── face_engine.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

## Installation

Python 3.11 is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit.

The Face Landmarker model downloads automatically
on first use.

## Deployment

Deploy using Streamlit Community Cloud:

1. Upload this repository to GitHub.
2. Visit https://share.streamlit.io/
3. Connect your GitHub account.
4. Select the FaceMesh-AI repository.
5. Set the main file to `app.py`.
6. Deploy the application.

## Modules

### Live Camera
Real-time webcam face tracking with mesh overlays.

### Photo Studio
Upload photographs and generate annotated face meshes.

### Video Studio
Process video frames and export annotated videos.

### Face Analysis
Inspect landmark coordinates and facial blendshape scores.

## Privacy

Image and video processing happens on the application server.
Uploaded media may be temporarily stored during processing.
Do not upload sensitive photographs to an untrusted deployment.

## Disclaimer

Facial landmarks and blendshape scores are model estimates.
They are not identity verification, medical diagnoses,
or definitive measurements of emotional state.

## License

MIT License
