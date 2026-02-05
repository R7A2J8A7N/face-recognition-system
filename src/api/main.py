from fastapi import FastAPI
from src.api.routes import recognize, enroll, health

app = FastAPI(
    title="Face Recognition Service",
    version="1.0.0"
)

app.include_router(health.router)
app.include_router(recognize.router)
# app.include_router(enroll.router)
