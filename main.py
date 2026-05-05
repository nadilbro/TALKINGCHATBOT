'''Initialising the AI bot'''

from fastapi import FastAPI, Request
from Router.edit_router import router as edit_router 
from Router.account_router import router as account_router 
from Router.voice_router import router as voice_router 
from Router.embed_router import router as embed_router
from fastapi.middleware.cors import CORSMiddleware
from Router.stripe_router import router as stripe_router
from SQL.db_init import init_db
from Router.stripe_business_router import router as stripe_business_router
from Router.Integrations.microsoft_oauth import router as microsoft_router
from Router.Integrations.google_oauth import router as google_router
app = FastAPI()


# CORS FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kannai.lovable.app",
        "https://7d6a4df9-b42e-4ddd-bedd-7fe6f8b70293.lovableproject.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# THEN routers
app.include_router(edit_router)
app.include_router(account_router)
app.include_router(stripe_router)
app.include_router(voice_router)
app.include_router(embed_router)   
app.include_router(stripe_business_router)
app.include_router(microsoft_router)

app.include_router(google_router)

@app.get("/")
def root():
    return {"status": "ok"}

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/healthz")
def healthz():
    return {"status": "healthy"}