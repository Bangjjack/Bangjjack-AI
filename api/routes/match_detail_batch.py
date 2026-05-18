# api/routes/match_detail_batch.py
# ==============================================================
# POST /match-detail-batch
#   1명(user) + 후보 N명(candidates) → 정렬된 Top K 매칭 상세 반환.
#   "홈 화면 추천 룸메이트" 같은 다대일 매칭에 사용.
#
#   /match-detail 을 N번 호출하는 것보다 효율적:
#     - predict_proba 를 N개 페어에 대해 한 번에 호출 (batch inference)
#     - SHAP 은 상위 top_k 후보에 대해서만 계산
# ==============================================================

from fastapi import APIRouter, Request

from api.schemas import (
    MatchDetailBatchRequest,
    MatchDetailBatchResponse,
    MatchDetailBatchItem,
)
from api.services.matching import compute_batch_details

router = APIRouter()


@router.post(
    "/match-detail-batch",
    response_model=MatchDetailBatchResponse,
    summary="매칭 상세 배치 (1:N, 정렬된 Top K)",
)
def match_detail_batch(req: MatchDetailBatchRequest, request: Request):
    """
    한 사용자와 후보자 목록을 받아 매칭률 내림차순으로 정렬된 결과를 반환합니다.

    - **top_k**: 반환할 상위 추천 수 (생략 시 전체 정렬 반환)
    - 응답의 `candidateIndex` 는 요청 `candidates` 배열에서의 원본 인덱스로,
      백엔드가 정렬 결과와 원본 후보를 매핑할 때 사용합니다.

    각 항목은 /match-detail 과 동일한 구조(matchRate, matchedFeatures,
    topInfluentialFeatures)에 정렬 정보(rank, candidateIndex)가 추가됩니다.
    """
    results = compute_batch_details(
        request.app.state.model,
        request.app.state.explainer,
        req.user,
        req.candidates,
        req.top_k,
    )
    return MatchDetailBatchResponse(
        ranked=[MatchDetailBatchItem(**r) for r in results]
    )
