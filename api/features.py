# api/features.py
# ==============================================================
# build_features() — UserProfile 두 개 → numpy 배열 (30개 피처)
# app.py / routes 에서 공통으로 import해서 사용
# ==============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # src/ 경로 추가

import numpy as np
from config import FEATURE_COLS, DIFF_COLS, MATCH_COLS

# DIFF_COLS 중 sleep_time은 순환 보정 필요 → 별도 처리
_DIFF_ORDINAL = [c for c in DIFF_COLS if c != "sleep_time"]


def _sleep_diff(a: float, b: float) -> float:
    """취침 시간 차이 — 자정 기준 순환 보정 (예: 23시 vs 01시 = 2시간)"""
    d = abs(a - b)
    return min(d, 24 - d)


def build_features(a: dict, b: dict) -> np.ndarray:
    """
    두 사람의 프로필 딕셔너리를 받아 30개 피처 배열 반환.

    Parameters
    ----------
    a, b : dict  (UserProfile.dict() 결과)

    Returns
    -------
    np.ndarray  shape (1, 30)  — model.predict_proba()에 바로 전달 가능
    """
    feats = {}

    # ── diff_10 ──────────────────────────────────────────────
    feats["diff_sleep_time"] = _sleep_diff(a["sleep_time"], b["sleep_time"])
    for col in _DIFF_ORDINAL:
        feats[f"diff_{col}"] = abs(a[col] - b[col])

    # ── match_20 ─────────────────────────────────────────────
    for col in MATCH_COLS:
        feats[f"match_{col}"] = int(a[col] == b[col])

    return np.array([[feats[c] for c in FEATURE_COLS]])
