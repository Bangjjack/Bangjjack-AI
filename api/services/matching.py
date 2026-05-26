# api/services/matching.py
# ==============================================================
# 매칭 점수 계산 공통 로직.
#
# /match-detail (페어), /match-detail-batch (1:N) 라우터가 공통으로 사용.
# 라우터는 입력/출력 스키마 변환만 담당하고 비즈니스 로직은 모두 여기로.
# ==============================================================

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from typing import List, Optional
import numpy as np

from api.schemas import RawProfile
from api.adapter import (
    raw_to_user_profile,
    DEFAULT_FEATURE_KEYS,
    ONE_HOT_MATCH_KEYS,
)
from api.features import build_features
from api.services.report import (
    evaluate_checklist,
    generate_summary_comment,
    generate_conversation_starters,
)
from config import FEATURE_COLS


# ── matchRate 보정 (logit 스케일링) ─────────────────────────
def logit_calibration(prob: float) -> float:
    """
    학습 데이터 특성상 모델 출력 확률이 양극단(≈0 또는 ≈1)에 몰리는
    문제를 logit 공간에서 선형으로 펼쳐 [0.10, 0.90] 범위로 매핑.

    matchRate = round(logit_calibration(prob) * 100) → 화면 점수 10~90.

    예시:
      prob=0.9997 → 0.90    prob=0.500  → 0.50
      prob=0.9800 → 0.69    prob=0.100  → 0.29
      prob=0.0027 → 0.20    prob=0.0001 → 0.10
    """
    LOW, HIGH   = 0.10, 0.90   # 표시 점수 하한/상한 (× 100 후 10~90점)
    LOGIT_RANGE = 8.0          # logit 클리핑 임계 (prob ≈ 0.9997 / 0.0003)

    p = max(1e-10, min(1 - 1e-10, prob))
    logit = math.log(p / (1 - p))
    logit = max(-LOGIT_RANGE, min(LOGIT_RANGE, logit))
    return LOW + (logit + LOGIT_RANGE) / (2 * LOGIT_RANGE) * (HIGH - LOW)


# ── matchedFeatures 필터 ────────────────────────────────────
def filter_matched_features(user_a: dict, feature_row: np.ndarray) -> List[str]:
    """
    match_* 피처 중 의미 있는 일치만 반환.
      ① match_* 접두어
      ② 값 = 1 (동일)
      ③ DEFAULT_FEATURE_KEYS(DB 미수집 항목) 제외
      ④ ONE_HOT_MATCH_KEYS(잠버릇·우선순위)는 "둘 다 1"인 경우만 인정
    """
    matched: List[str] = []
    for col, val in zip(FEATURE_COLS, feature_row):
        if not col.startswith("match_"):
            continue
        if int(val) != 1:
            continue
        if col in DEFAULT_FEATURE_KEYS:
            continue
        if col in ONE_HOT_MATCH_KEYS:
            underlying = col[len("match_"):]
            if int(user_a.get(underlying, 0)) != 1:
                continue   # 둘 다 0(없음)인 매칭은 제외
        matched.append(col)
    return matched


# ── topInfluentialFeatures (SHAP 기반) ─────────────────────
def top_influential_features(sv_row: np.ndarray, k: int = 3) -> List[str]:
    """
    한 샘플의 SHAP 값(길이 30) → 영향력 절댓값 Top K 피처 키 반환.
    DEFAULT_FEATURE_KEYS는 제외 (모델이 사실상 무시하는 항목).
    """
    ranked = sorted(
        (
            (col, val) for col, val in zip(FEATURE_COLS, sv_row.tolist())
            if col not in DEFAULT_FEATURE_KEYS
        ),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [name for name, _ in ranked[:k]]


# ── /match-detail 진입점 (단건 페어) ───────────────────────
def compute_pair_detail(model, explainer,
                        user_a_raw: RawProfile,
                        user_b_raw: RawProfile) -> dict:
    """
    한 쌍에 대한 풀 매칭 리포트 dict 반환.

    Keys:
      - matchRate (int)
      - matchedFeatures     (List[dict] — key/label/description)
      - mismatchedFeatures  (List[dict] — key/label/description/advice)
      - topInfluentialFeatures (List[str] — SHAP 영향력 Top 3 key)
      - summaryComment      (dict — positive/caution)
      - counts              (dict — matched/mismatched/total)
    """
    user_a = raw_to_user_profile(user_a_raw)
    user_b = raw_to_user_profile(user_b_raw)
    X = build_features(user_a, user_b)

    prob = float(model.predict_proba(X)[0][1])
    match_rate = int(round(logit_calibration(prob) * 100))

    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv_row = sv[0]   # shape (30,)

    # 체크리스트 항목별 평가 (8개 항목 → matched + mismatched)
    matched, mismatched = evaluate_checklist(user_a_raw, user_b_raw)

    # 자연어 종합 코멘트 (brief / positive / caution)
    summary = generate_summary_comment(matched, mismatched, match_rate)

    # SHAP 영향력 Top 키 → 대화 시작 문구 변환
    top_keys = top_influential_features(sv_row)
    starters = generate_conversation_starters(top_keys)

    return {
        "matchRate":              match_rate,
        "matchedFeatures":        matched,
        "mismatchedFeatures":     mismatched,
        "topInfluentialFeatures": top_keys,
        "conversationStarters":   starters,
        "summaryComment":         summary,
        "counts": {
            "matched":    len(matched),
            "mismatched": len(mismatched),
            "total":      len(matched) + len(mismatched),
        },
    }


# ── /match-detail-batch 진입점 (1:N 배치) ─────────────────
def compute_batch_details(model, explainer,
                          user_raw: RawProfile,
                          candidates_raw: List[RawProfile],
                          top_k: Optional[int] = None) -> List[dict]:
    """
    1명 + 후보 N명 → 매칭률 내림차순 정렬된 dict 리스트.

    성능 최적화: 2-pass 구조
      [1패스] 모든 후보 피처 한 번에 쌓아 batch predict_proba (SHAP 미포함)
      [2패스] 상위 top_k 후보에 대해서만 SHAP 계산 + 디테일 빌드

    SHAP 비용이 추론 대비 훨씬 크기 때문에, top_k가 작을수록 효과 큼.

    반환 dict keys: rank, candidateIndex, matchRate,
                   matchedFeatures, topInfluentialFeatures
    """
    if not candidates_raw:
        return []

    # 1) 사용자/후보 변환
    user_dict       = raw_to_user_profile(user_raw)
    candidate_dicts = [raw_to_user_profile(c) for c in candidates_raw]

    # 2) 모든 페어 피처 배열 한 번에 생성 — shape (N, 30)
    feature_matrix = np.vstack([
        build_features(user_dict, c_dict)[0] for c_dict in candidate_dicts
    ])

    # 3) 배치 추론 (predict_proba 1회 호출로 N개 prob 동시 계산)
    probs = model.predict_proba(feature_matrix)[:, 1]

    # 4) 내림차순 정렬 + top_k 컷
    sorted_indices = np.argsort(probs)[::-1]
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]

    # 5) 선택된 후보만 SHAP 일괄 계산
    selected_X = feature_matrix[sorted_indices]   # (k, 30)
    sv_all = explainer.shap_values(selected_X)
    if isinstance(sv_all, list):
        sv_all = sv_all[1]
    # sv_all shape: (k, 30) — i번째 행이 sorted_indices[i] 후보의 SHAP

    # 6) 응답 빌드 — 각 후보에 대해 체크리스트 평가 + 자연어 코멘트 포함
    results: List[dict] = []
    for rank, idx in enumerate(sorted_indices, start=1):
        idx_int    = int(idx)
        match_rate = int(round(logit_calibration(float(probs[idx_int])) * 100))

        matched, mismatched = evaluate_checklist(user_raw, candidates_raw[idx_int])
        summary  = generate_summary_comment(matched, mismatched, match_rate)
        top_keys = top_influential_features(sv_all[rank - 1])
        starters = generate_conversation_starters(top_keys)

        results.append({
            "rank":                   rank,
            "candidateIndex":         idx_int,
            "matchRate":              match_rate,
            "matchedFeatures":        matched,
            "mismatchedFeatures":     mismatched,
            "topInfluentialFeatures": top_keys,
            "conversationStarters":   starters,
            "summaryComment":         summary,
            "counts": {
                "matched":    len(matched),
                "mismatched": len(mismatched),
                "total":      len(matched) + len(mismatched),
            },
        })
    return results
