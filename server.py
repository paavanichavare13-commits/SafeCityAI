from fastapi import FastAPI, UploadFile, File
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import tempfile
import os

app = FastAPI(title="SafeCityAI API")

WORKSPACE = "paavani-chavare"
WORKFLOW_ID = "helmet-and-license-plate-vhelmet-and-license-plate-hfdzo-lzojk-1-yolo26s-t1-logic"

API_KEY = os.environ["ROBOFLOW_API_KEY"]

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
).configure(
    InferenceConfiguration(api_key_transport="header")
)


@app.get("/")
def home():
    return {"message": "SafeCityAI API is running"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    contents = await file.read()

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    ) as temp:

        temp.write(contents)
        temp_path = temp.name

    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE,
            workflow_id=WORKFLOW_ID,
            images={
                "image": temp_path
            },
            use_cache=False
        )

        return {
            "success": True,
            "result": result
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)