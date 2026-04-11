# api/routes/matches.py
# ==============================================================
# POST /top-matches  —  후보자 중 가장 잘 맞는 K명 반환
# ==============================================================

from fastapi import APIRouter, Request
from api.schemas import TopMatchRequest, TopMatchResponse, MatchResult
from api.features import build_features

router = APIRouter()


@router.post("/top-matches", response_model=TopMatchResponse, summary="상위 매칭 추천")
def top_matches(req: TopMatchRequest, request: Request):
    """
    입력 유저와 후보자 목록을 비교해 호환 확률이 높은 순서로 상위 K명을 반환합니다.

    - **top_k**: 반환할 추천 수 (기본값 5)
    - **rank**: 1이 가장 호환 확률이 높음
    """
    model = request.app.state.model
    a     = req.user.dict()

    # 모든 후보자와 확률 계산
    results = []
    for idx, candidate in enumerate(req.candidates):
        X    = build_features(a, candidate.dict())
        prob = float(model.predict_proba(X)[0][1])
        results.append({"candidate_index": idx, "probability": round(prob, 4)})

    # 확률 내림차순 정렬 → 상위 K개
    results.sort(key=lambda x: x["probability"], reverse=True)
    top = results[: req.top_k]
    for rank, r in enumerate(top, 1):
        r["rank"] = rank

    return TopMatchResponse(
        top_matches=[MatchResult(**r) for r in top]
    )
