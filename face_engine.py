
import os
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "face_landmarker.task"


def ensure_model():
    MODEL_DIR.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        temp_path = MODEL_PATH.with_suffix(".download")

        try:
            urllib.request.urlretrieve(MODEL_URL, temp_path)
            temp_path.replace(MODEL_PATH)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the Face Landmarker model."
            )

    return str(MODEL_PATH)


def create_landmarker(mode="IMAGE", faces=2):
    model = ensure_model()

    running_mode = getattr(
        mp.tasks.vision.RunningMode, mode
    )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=model
        ),
        running_mode=running_mode,
        num_faces=faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    return mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )


def render_mesh(
    image,
    result,
    mesh=True,
    contours=True,
    points=False,
):
    output = image.copy()
    height, width = output.shape[:2]

    connections = mp.tasks.vision.FaceLandmarksConnections

    for landmarks in result.face_landmarks:

        if mesh:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                image=output,
                landmark_list=landmarks,
                connections=connections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=
                    mp.tasks.vision.drawing_utils.DrawingSpec(
                        color=(255, 180, 0),
                        thickness=1,
                        circle_radius=1,
                    ),
            )

        if contours:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                image=output,
                landmark_list=landmarks,
                connections=connections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=
                    mp.tasks.vision.drawing_utils.DrawingSpec(
                        color=(0, 255, 255),
                        thickness=2,
                        circle_radius=1,
                    ),
            )

        if points:
            for landmark in landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(
                        output,
                        (x, y),
                        1,
                        (0, 255, 120),
                        -1,
                    )

    return output


def analyze_image(image, faces=2):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    with create_landmarker("IMAGE", faces) as detector:
        result = detector.detect(mp_image)

    return result


def get_expression_data(result, face_index=0):
    if not result.face_blendshapes:
        return {}

    if face_index >= len(result.face_blendshapes):
        return {}

    categories = result.face_blendshapes[face_index].categories

    return {
        item.category_name: float(item.score)
        for item in categories
    }


def get_landmark_data(result, face_index=0):
    if face_index >= len(result.face_landmarks):
        return []

    return [
        {
            "index": index,
            "x": round(float(point.x), 6),
            "y": round(float(point.y), 6),
            "z": round(float(point.z), 6),
        }
        for index, point in enumerate(
            result.face_landmarks[face_index]
        )
    ]


def process_video(
    uploaded_file,
    mesh=True,
    contours=True,
    points=False,
    faces=2,
):
    temp_input = Path("input_video.mp4")
    temp_output = Path("facemesh_output.mp4")

    temp_input.write_bytes(uploaded_file.getvalue())

    capture = cv2.VideoCapture(str(temp_input))

    if not capture.isOpened():
        raise RuntimeError("Unable to open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Unable to create output video.")

    processed = 0
    start_time = time.time()

    try:
        with create_landmarker("VIDEO", faces) as detector:

            while True:
                success, frame = capture.read()

                if not success:
                    break

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = int(
                    processed * 1000 / fps
                )

                result = detector.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )

                output = render_mesh(
                    frame,
                    result,
                    mesh,
                    contours,
                    points,
                )

                writer.write(output)
                processed += 1

    finally:
        capture.release()
        writer.release()
        temp_input.unlink(missing_ok=True)

    return {
        "path": str(temp_output),
        "processed": processed,
        "total": total_frames,
        "elapsed": round(time.time() - start_time, 2),
    }


def calculate_fps(start, frames):
    elapsed = max(time.time() - start, 0.001)
    return round(frames / elapsed, 1)
