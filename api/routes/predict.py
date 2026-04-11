# api/routes/predict.py
# ==============================================================
# POST /predict  —  두 사람의 호환 확률 반환
# ==============================================================

from fastapi import APIRouter, Request
from api.schemas import PairRequest, PredictResponse
from api.features import build_features

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, summary="호환 예측")
def predict(req: PairRequest, request: Request):
    """
    두 사람의 프로필을 입력받아 호환 여부와 확률을 반환합니다.

    - **compatible**: 1 = 호환, 0 = 비호환
    - **probability**: 호환 확률 (0.0 ~ 1.0)
    """
    model = request.app.state.model

    X    = build_features(req.user_a.dict(), req.user_b.dict())
    prob = float(model.predict_proba(X)[0][1])

    return PredictResponse(
        compatible=int(prob >= 0.5),
        probability=round(prob, 4),
    )
