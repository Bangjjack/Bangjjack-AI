# api/routes/explain.py
# ==============================================================
# POST /explain  —  SHAP 기반 피처 기여도 반환
# ==============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fastapi import APIRouter, Request
from api.schemas import PairRequest, ExplainResponse, FeatureContribution
from api.features import build_features
from config import FEATURE_COLS

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse, summary="SHAP 설명")
def explain(req: PairRequest, request: Request):
    """
    두 사람의 프로필을 입력받아 호환 판단 근거(SHAP 값 상위 5개)를 반환합니다.

    - **top_features**: 호환 판단에 가장 영향을 준 피처 목록
      - shap_value > 0 : 호환에 유리하게 작용
      - shap_value < 0 : 비호환에 유리하게 작용
    """
    model     = request.app.state.model
    explainer = request.app.state.explainer

    X    = build_features(req.user_a.dict(), req.user_b.dict())
    prob = float(model.predict_proba(X)[0][1])

    # SHAP 값 계산
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]   # binary: class=1 (호환) 기준 shap values
    sv = sv[0]       # shape (30,)

    # 절댓값 기준 상위 5개 추출
    top5 = sorted(
        zip(FEATURE_COLS, sv.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]

    return ExplainResponse(
        compatible=int(prob >= 0.5),
        probability=round(prob, 4),
        top_features=[
            FeatureContribution(feature=f, shap_value=round(v, 4))
            for f, v in top5
        ],
    )
