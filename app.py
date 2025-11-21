from fastapi import FastAPI, Request
import uvicorn
import os
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response

from textSummarizer.pipelines.prediction import PredictionPipeline

from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")
predictor = PredictionPipeline()


class SummarizationRequest(BaseModel):
    text: str


@app.get("/", tags=["web"])
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "summary": None, "input_text": ""}
    )


@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training Successful")
    
    except Exception as e:
        return Response(f"Error Occurred! {e}") 
    

@app.post("/predict")
async def predict_route(request: SummarizationRequest):
    try:
        summary = predictor.predict(request.text)
        return {"summary": summary}
    
    except Exception as e:
        return Response(f"Error Occurred! {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
