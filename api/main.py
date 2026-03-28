"""
main.py
-------
FastAPI application entry point for the OilGasAI backend.
Serves OilGasAI-Model-Alpha via REST API with chat, compliance, and sensor endpoints.
"""

import os
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from inference.load_model import get_inference_backend
from rag.rag_pipeline import RAGPipeline
from api.routes import chat, compliance, sensors

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ── App state ────────────────────────────────────────────────────────────────

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and RAG pipeline on startup."""
    logger.info("Starting OilGasAI API — loading OilGasAI-Model-Alpha...")
    app_state["backend"] = get_inference_backend()
    app_state["rag"] = RAGPipeline()
    logger.info("OilGasAI API ready.")
    yield
    logger.info("Shutting down OilGasAI API.")
    app_state.clear()


# ── App init ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OilGasAI — OilGasAI Model Alpha API",
    description=(
        "Domain-expert AI for oil and gas operations. "
        "Powered by OilGasAI-Model-Alpha — QLoRA fine-tuned Llama 3.1 70B. "
        "Specializes in EPA compliance, methane monitoring, LDAR, and predictive maintenance."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ─────────────────────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not API_KEY:
        return  # no auth required if API_KEY not set
    if not credentials or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Dependency injection ──────────────────────────────────────────────────────

def get_backend():
    return app_state["backend"]

def get_rag():
    return app_state["rag"]


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    compliance.router,
    prefix="/compliance",
    tags=["Compliance"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    sensors.router,
    prefix="/sensors",
    tags=["Sensors"],
    dependencies=[Depends(verify_api_key)],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "model": "OilGasAI-Model-Alpha",
        "backend": app_state.get("backend", {}).get("mode", "loading"),
        "platform": "https://innovations1-spec.github.io/oilgasai/",
    }


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Welcome to OilGasAI — OilGasAI-Model-Alpha API",
        "docs": "/docs",
        "platform": "https://innovations1-spec.github.io/oilgasai/",
        "model": "https://huggingface.co/OilgasAI/OilGasAI-Model-Alpha",
    }
