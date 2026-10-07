import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from data_layer import ProductRepository, OrderRepository
from application_layer import QueryService, SpeechService, ImageService, SearchService, RankingService
from presentation_layer import SearchUI

app = FastAPI(title="ShopLens Multimodal Search")

# Setup static files and templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Initialize Data Layer
product_repo = ProductRepository(os.path.join(BASE_DIR, "data", "products.json"))
order_repo = OrderRepository(os.path.join(BASE_DIR, "data", "orders.json"))

# Initialize Application Layer
query_service = QueryService()
speech_service = SpeechService()
image_service = ImageService()
search_service = SearchService(product_repo)
ranking_service = RankingService()

# Initialize Presentation Controller
search_ui = SearchUI(
    query_service=query_service,
    speech_service=speech_service,
    image_service=image_service,
    search_service=search_service,
    ranking_service=ranking_service,
    order_repo=order_repo
)

class SearchRequest(BaseModel):
    text: Optional[str] = ""
    image_name: Optional[str] = None
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    wt: Optional[float] = None
    wi: Optional[float] = None

class VoiceRequest(BaseModel):
    simulated_transcript: Optional[str] = "Find me some white shoes."
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None

class ImageRequest(BaseModel):
    image_name: str
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None

class MultimodalRequest(BaseModel):
    text: str
    image_name: str
    wt: Optional[float] = 0.5
    wi: Optional[float] = 0.5
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"query_text": ""})

@app.post("/api/search")
async def api_search(req: SearchRequest):
    if req.image_name and req.text:
        wt = req.wt if req.wt is not None else 0.5
        wi = req.wi if req.wi is not None else 0.5
        return search_ui.handle_multimodal_search(req.text, req.image_name, wt, wi, req.category, req.min_price, req.max_price)
    elif req.image_name:
        return search_ui.handle_image_search(req.image_name, req.category, req.min_price, req.max_price)
    else:
        return search_ui.handle_text_search(req.text or "", req.category, req.min_price, req.max_price)

@app.post("/api/search_voice")
async def api_search_voice(req: VoiceRequest):
    return search_ui.handle_voice_search(req.simulated_transcript, req.category, req.min_price, req.max_price)

@app.post("/api/search_image")
async def api_search_image(req: ImageRequest):
    return search_ui.handle_image_search(req.image_name, req.category, req.min_price, req.max_price)

@app.post("/api/search_multimodal")
async def api_search_multimodal(req: MultimodalRequest):
    wt = req.wt if req.wt is not None else 0.5
    wi = req.wi if req.wi is not None else 0.5
    return search_ui.handle_multimodal_search(req.text, req.image_name, wt, wi, req.category, req.min_price, req.max_price)

@app.get("/api/order/{order_id}")
async def api_order(order_id: str):
    return search_ui.handle_order_lookup(order_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
