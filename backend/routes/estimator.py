# Routen fuer Estimator-Konfiguration (/api/estimator/*)

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from routes.occupancy import estimator

router = APIRouter()


class BaselineRequest(BaseModel):
    gas_resistance: float = 200000


@router.get("/api/estimator/status")
def api_estimator_status():
    return {"success": True, "data": estimator.get_status()}


@router.post("/api/estimator/baseline")
def api_set_baseline(body: BaselineRequest):
    estimator.set_baseline(gas_resistance=body.gas_resistance)
    return {"success": True, "message": "VOC-Baseline gesetzt",
            "data": estimator.get_status()}
