from custom_encoders import custom_jsonable_encoder  # This must be the first import
from fastapi import FastAPI
from routes import router
from fastapi.responses import JSONResponse

app = FastAPI()
app.include_router(router)