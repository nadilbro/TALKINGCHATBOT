'''Initialising the AI bot'''

from fastapi import FastAPI, Request
from Router.chat_router import router as chat_router
from Router.edit_router import router as edit_router 
from Router.startup_router import router as startup_router 
from Router.voice_router import router as voice_router 
from fastapi.middleware.cors import CORSMiddleware
from SQL.db_init import init_db
app = FastAPI()



app.include_router(chat_router)
app.include_router(edit_router) 
app.include_router(startup_router) 
app.include_router(voice_router) 


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"➡️ {request.method} {request.url.path}", flush=True)
    return await call_next(request)

origins = [
    "https://bubbleworks.com.au",
    "https://www.bubbleworks.com.au",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://b5b11fb7-af8d-4fae-b773-1bf1035a8d71.lovableproject.com",
    "https://id-preview--b5b11fb7-af8d-4fae-b773-1bf1035a8d71.lovable.app",
    "https://trait-tinker-lab.lovable.app"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],          # IMPORTANT: includes OPTIONS
    allow_headers=["*"],          # IMPORTANT: includes Accept / Content-Type
)


@app.get("/")
def root():
    return {"status": "ok"}

@app.on_event("startup")
def on_startup():
    init_db()

