'''Initialising the AI bot'''

from fastapi import FastAPI
from Router.chat_router import router as chat_router
from fastapi.middleware.cors import CORSMiddleware
from SQL.db_init import init_db
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://b5b11fb7-af8d-4fae-b773-1bf1035a8d71.lovableproject.com",
        "https://id-preview--b5b11fb7-af8d-4fae-b773-1bf1035a8d71.lovable.app",
        # add your published domain later too
    ],
    allow_credentials=True,
    allow_methods=["*"],        # includes OPTIONS for preflight
    allow_headers=["*"],        # MUST include Authorization + Content-Type
)


app.include_router(chat_router)

@app.get("/")
def root():
    return {"status": "ok"}

@app.on_event("startup")
def on_startup():
    init_db()