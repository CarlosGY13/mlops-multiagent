from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers import part1, part2, part3, system

settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.1.0")

app.include_router(system.router)
app.include_router(part1.router)
app.include_router(part2.router)
app.include_router(part3.router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def root():
    return FileResponse("app/static/index.html")
