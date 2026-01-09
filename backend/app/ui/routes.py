"""UI routes for serving HTML templates."""
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Setup templates
templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "page": "dashboard"}
    )


@router.get("/incidents", response_class=HTMLResponse)
async def incidents_list(request: Request):
    """Incidents list page."""
    return templates.TemplateResponse(
        "incidents.html",
        {"request": request, "page": "incidents"}
    )


@router.get("/incidents/{incident_id}", response_class=HTMLResponse)
async def incident_detail(request: Request, incident_id: str):
    """Incident detail page."""
    return templates.TemplateResponse(
        "incident_detail.html",
        {"request": request, "page": "incidents", "incident_id": incident_id}
    )


@router.get("/demo", response_class=HTMLResponse)
async def demo_control(request: Request):
    """Demo control panel."""
    return templates.TemplateResponse(
        "demo.html",
        {"request": request, "page": "demo"}
    )

