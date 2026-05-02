import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline.routers import hardware, runs, sse, templates
from pipeline.settings import settings

app = FastAPI(title="mad-lab-train pipeline server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:18800", "http://localhost:18810", "http://127.0.0.1:18800"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(sse.router)
app.include_router(templates.router)
app.include_router(hardware.router)


def main() -> None:
    uvicorn.run("pipeline.server:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
