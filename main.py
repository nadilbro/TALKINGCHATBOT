'''Initialising the AI bot'''

from fastapi import FastAPI, Request
from Router.edit_router import router as edit_router 
from Router.startup_router import router as startup_router 
from Router.voice_router import router as voice_router 
from fastapi.middleware.cors import CORSMiddleware
from SQL.db_init import init_db
app = FastAPI()


app.include_router(edit_router) 
app.include_router(startup_router) 
app.include_router(voice_router) 



app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],          # IMPORTANT: includes OPTIONS
    allow_headers=["*"],          # IMPORTANT: includes Accept / Content-Type
)


@app.get("/")
def root():
    return {"status": "ok"}

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/healthz")
def healthz():
    return {"status": "healthy"}