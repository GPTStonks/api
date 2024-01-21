from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .agent import run_agent_in_background
from .auth_routers import tokens
from .constants import API_DESCRIPTION
from .initialization import init_api
from .models import AppData

app_data = AppData()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize everything
    init_api(app_data=app_data)
    yield
    # Nothing to do when releasing


app = FastAPI(
    title="GPTStonks Chat CE API",
    description=API_DESCRIPTION,
    version="0.0.1",
    contact={
        "name": "GPTStonks, part of Mitheithel",
        "email": "contact@mitheithel.com",
    },
    license_info={
        "name": "Apache 2.0",
        "identifier": "MIT",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tokens.router)


@app.post("/process_query_async")
async def process_query_async(request: Request):
    """Asynchronous endpoint to start processing the given query. The processing runs in the
    background and the result is eventually returned.

    Args:
        request (Request): FastAPI request object containing the query to be processed.

    Returns:
        dict: containing the response.
    """
    data = await request.json()
    query = data.get("query")

    return await run_agent_in_background(query=query, app_data=app_data)
