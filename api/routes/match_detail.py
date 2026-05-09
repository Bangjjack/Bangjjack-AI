# api/routes/match_detail.py
# ==============================================================
# POST /match-detail
#   백엔드 통합용 단일 엔드포인트.
#   DB ENUM 그대로 받아 매칭률 + 일치/영향 피처를 반환.
#
# 응답 키들은 백엔드에서 한국어 라벨로 매핑해 사용.
# ==============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fastapi import APIRouter, Request

from api.schemas import MatchDetailRequest, MatchDetailResponse
from api.adapter import (
    raw_to_user_profile,
    DEFAULT_FEATURE_KEYS,
    ONE_HOT_MATCH_KEYS,
)
from api.features import build_features
from config import FEATURE_COLS

router = APIRouter()


@router.post(
    "/match-detail",
    response_model=MatchDetailResponse,
    summary="매칭 상세 (백엔드 통합용)",
)
def match_detail(req: MatchDetailRequest, request: Request):
    """
    DB ENUM 입력 → 매칭률 + 일치/영향 피처 반환.

    - **matchRate**: 0~100 정수 (probability × 100 반올림)
    - **matchedFeatures**: `match_*` 피처 중 두 사용자가 일치한 항목 키 목록
        - 백엔드에서 라벨 매핑하여 `matchedAttributes`로 사용
    - **topInfluentialFeatures**: SHAP 절댓값 기준 영향력 Top 3 피처 키
        - 백엔드에서 라벨 매핑하여 `recommendedTopics`로 사용

    응답 키 예시:
    - `match_smoking`, `match_gender`, `match_clean_freq`
    - `diff_sleep_time`, `diff_clean_freq`
    """
    model     = request.app.state.model
    explainer = request.app.state.explainer

    # 1) DB ENUM → AI 내부 형식(dict) 변환
    user_a = raw_to_user_profile(req.user_a)
    user_b = raw_to_user_profile(req.user_b)

    # 2) 30개 피처 배열 생성 + 모델 추론
    X = build_features(user_a, user_b)
    prob = float(model.predict_proba(X)[0][1])

    # 2-1) matchRate 보정 (sqrt 스케일)
    #   학습 데이터 특성상 모델 출력이 양극단(≈0 또는 ≈1)에 몰림.
    #   raw prob을 그대로 % 환산하면 비호환 케이스가 0%로 표시되어
    #   UX가 어색하므로 sqrt(prob)로 살짝 완화.
    #   - prob=1.000 → 100,  prob=0.94 → 97
    #   - prob=0.50 → 71,    prob=0.01 → 10
    #   - prob=0.003 → 5,    prob=0.0001 → 1
    display_score = prob ** 0.5

    # 3) SHAP 값 계산 (class=1 기준)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]   # shape: (30,)

    # 4) matchedFeatures: match_* 중 값이 1이면서 의미 있는 매칭만 남김.
    #    필터 단계:
    #      ① match_* 접두어
    #      ② build_features 결과가 1 (동일)
    #      ③ DEFAULT_FEATURE_KEYS(DB 미수집/기본값) 제외
    #      ④ one-hot 인코딩 피처(habit_*, prio*_*)는 "둘 다 1"인 경우만 인정
    #         (둘 다 0=0인 false positive 매칭 제거)
    feature_values = X[0]
    matched_features = []
    for col, val in zip(FEATURE_COLS, feature_values):
        if not col.startswith("match_"):
            continue
        if int(val) != 1:
            continue
        if col in DEFAULT_FEATURE_KEYS:
            continue
        if col in ONE_HOT_MATCH_KEYS:
            # 원본 dict 키는 'match_' 접두어를 제거한 이름
            # 예: 'match_habit_snore' → 'habit_snore'
            underlying = col[len("match_"):]
            if int(user_a.get(underlying, 0)) != 1:
                continue   # 둘 다 0(없음)인 매칭은 제외
        matched_features.append(col)

    # 5) topInfluentialFeatures: SHAP 절댓값 Top 3
    #    (기본값으로 채워진 피처는 모델 영향도가 거의 0이라 자연 제외되지만
    #     안전하게 명시적으로도 필터링)
    ranked = sorted(
        (
            (col, val) for col, val in zip(FEATURE_COLS, sv.tolist())
            if col not in DEFAULT_FEATURE_KEYS
        ),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_influential = [name for name, _ in ranked[:3]]

    return MatchDetailResponse(
        matchRate=int(round(display_score * 100)),
        matchedFeatures=matched_features,
        topInfluentialFeatures=top_influential,
    )
