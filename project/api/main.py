"""
FastAPI wrapper: https://.../compare?cbb_url=<sports-ref-url>
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from project.compare.engine import run_all
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI(title="CBB → NBA Comparator")

# ── 1.  API routes FIRST ────────────────────────────────────────────
@app.get("/compare")
async def compare(
    cbb_url: str = Query(..., description="Sports-Reference college player URL")
):
    try:
        return run_all(cbb_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── 2.  CORS (optional, keep if you need it) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 3.  STATIC FILES LAST  (won't mask /compare now) ────────────────
BASE_DIR = Path(__file__).resolve().parent        # project/api
app.mount(
    "/", StaticFiles(directory=BASE_DIR / "Static", html=True), name="static"
)

# ── 4.  Local dev entry-point (Render ignores this) ────────────────
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(
        "project.api.main:app",  # dotted path from repo root
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
