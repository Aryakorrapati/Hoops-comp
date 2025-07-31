"""
FastAPI wrapper:  http://localhost:8000/compare?cbb_url=<full-sports-ref-url>
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from project.compare.engine import run_all
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="CBB → NBA Comparator")
app.mount("/static", StaticFiles(directory="api/static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],          # not just ["GET"]
    allow_headers=["*"],
)

@app.get("/compare")
async def compare(cbb_url: str = Query(..., description="Sports-Reference college player URL")):
    try:
        return run_all(cbb_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Dev entry-point:  python -m api.main
if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
