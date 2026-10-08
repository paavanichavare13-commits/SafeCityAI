import cv2
import os
import tempfile

from inference_sdk import InferenceHTTPClient, InferenceConfiguration


# =========================
# ROBoflow API KEY
# =========================

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY is not set. "
        "Set it in Terminal before running the program."
    )


# =========================
# ROBoflow CLIENT
# =========================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)


# =========================
# PROJECT SETTINGS
# =========================

WORKSPACE = "paavani-chavare"

WORKFLOW_ID = (
    "helmet-and-license-plate-"
    "vhelmet-and-license-plate-hfdzo-lzojk-1-yolo26s-t1-logic"
)

INPUT_VIDEO = "traffic.mp4"
OUTPUT_VIDEO = "annotated_traffic.mp4"


# =========================
# OPEN VIDEO
# =========================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open traffic.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# =========================
# OUTPUT VIDEO
# =========================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# =========================
# PROCESS VIDEO
# =========================

frame_number = 0

print("🚀 Starting SafeCityAI video inference...")
print()


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Save frame temporarily
    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as temp:

        temp_path = temp.name

    cv2.imwrite(temp_path, frame)

    try:

        # Send frame to Roboflow
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={
                "image": temp_path
            },
            use_cache=False
        )

        # =========================
        # GET PREDICTIONS
        # =========================

        if isinstance(result, list):
            result = result[0] if result else {}

        predictions = result.get("predictions", [])

        # Some workflows may return predictions inside a list/dict
        if isinstance(predictions, dict):
            predictions = predictions.get("predictions", [])

        if not isinstance(predictions, list):
            predictions = []

        # =========================
        # DRAW BOXES
        # =========================

        for prediction in predictions:

            x = prediction.get("x")
            y = prediction.get("y")
            w = prediction.get("width")
            h = prediction.get("height")

            confidence = prediction.get(
                "confidence",
                0
            )

            class_name = prediction.get(
                "class",
                "object"
            )

            if None in (x, y, w, h):
                continue

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)

            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Keep box inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"{class_name} {confidence:.0%}"

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        print(
            f"Frame {frame_number}: "
            f"{len(predictions)} detections"
        )

    finally:

        os.remove(temp_path)

    out.write(frame)


# =========================
# FINISH
# =========================

cap.release()
out.release()

print()
print("===================================")
print("✅ VIDEO INFERENCE COMPLETE")
print("===================================")
print(f"🎥 Output: {OUTPUT_VIDEO}")
print(f"📊 Frames processed: {frame_number}")
