# api/routes/match_detail.py
# ==============================================================
# POST /match-detail
#   백엔드 통합용 페어 매칭 엔드포인트.
#   DB ENUM 두 사람 그대로 받아 매칭률 + 일치/영향 피처 반환.
#
# 비즈니스 로직은 api/services/matching.py 에 위임.
# 후보 N명 정렬·추천이 필요한 경우 /match-detail-batch 사용.
# ==============================================================

from fastapi import APIRouter, Request

from api.schemas import MatchDetailRequest, MatchDetailResponse
from api.services.matching import compute_pair_detail

router = APIRouter()


@router.post(
    "/match-detail",
    response_model=MatchDetailResponse,
    summary="매칭 상세 (페어)",
)
def match_detail(req: MatchDetailRequest, request: Request):
    """
    두 사용자의 DB ENUM을 받아 매칭률 + 일치/영향 피처를 반환합니다.

    - **matchRate**: 0~100 정수 (logit 보정 적용, 보통 10~90 범위)
    - **matchedFeatures**: 의미 있게 일치한 항목 키 목록
    - **topInfluentialFeatures**: SHAP 영향력 Top 3 피처 키

    1명 + 후보 N명 정렬·추천이 필요하면 /match-detail-batch 를 사용하세요.
    """
    detail = compute_pair_detail(
        request.app.state.model,
        request.app.state.explainer,
        req.user_a,
        req.user_b,
    )
    return MatchDetailResponse(**detail)
