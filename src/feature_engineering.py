# src/feature_engineering.py
# ==============================================================
# 2단계: bangjjak_processed.csv → bangjjak_pairs_v2.csv
# 실행: python src/feature_engineering.py
# ==============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from itertools import combinations
from config import (
    PROC_PATH, PAIR_PATH,
    DIFF_COLS, MATCH_COLS, FEATURE_COLS,
    CI_COLS, CI_WEIGHTS, COMPAT_THRESHOLD,
)


def _sleep_diff_vec(sl_a: np.ndarray, sl_b: np.ndarray) -> np.ndarray:
    """취침 시간 차이 벡터화 — 자정 기준 순환 보정"""
    raw = np.abs(sl_a - sl_b)
    return np.minimum(raw, 24 - raw)


def make_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """모든 응답자 쌍(C(n,2))에 대해 diff/match/ci 피처 계산"""
    n = len(df)
    arr_a, arr_b = map(np.array, zip(*combinations(range(n), 2)))
    print(f"  C({n}, 2) = {len(arr_a):,} 쌍 생성 중...")

    pairs = pd.DataFrame({"idx_a": arr_a, "idx_b": arr_b})

    # 배열 캐싱 (속도 최적화)
    v = {col: df[col].values for col in df.columns}

    # ── diff_10 ──────────────────────────────────────────────
    pairs["diff_sleep_time"] = _sleep_diff_vec(v["sleep_time"][arr_a], v["sleep_time"][arr_b])
    for col in [c for c in DIFF_COLS if c != "sleep_time"]:
        pairs[f"diff_{col}"] = np.abs(v[col][arr_a] - v[col][arr_b])

    # ── match_20 ─────────────────────────────────────────────
    for col in MATCH_COLS:
        pairs[f"match_{col}"] = (v[col][arr_a] == v[col][arr_b]).astype(int)

    return pairs


def compute_ci(pairs: pd.DataFrame, v: dict) -> pd.DataFrame:
    """ci_(9개) 충돌 지표 계산 — 레이블 생성 전용"""
    arr_a = pairs["idx_a"].values
    arr_b = pairs["idx_b"].values

    sl_a = v["sleep_time"][arr_a];      sl_b = v["sleep_time"][arr_b]
    cf_a = v["clean_freq"][arr_a];      cf_b = v["clean_freq"][arr_b]
    hs_a = v["habit_snore"][arr_a];     hs_b = v["habit_snore"][arr_b]
    ns_a = v["noise_sensitivity"][arr_a]; ns_b = v["noise_sensitivity"][arr_b]
    nc_a = v["night_call"][arr_a];      nc_b = v["night_call"][arr_b]
    td_a = v["tolerance"][arr_a];       td_b = v["tolerance"][arr_b]
    sm_a = v["smoking"][arr_a];         sm_b = v["smoking"][arr_b]
    pi_a = v["privacy_importance"][arr_a]; pi_b = v["privacy_importance"][arr_b]
    ei_a = v["eating_in_room"][arr_a];  ei_b = v["eating_in_room"][arr_b]
    ts_a = v["temp_enc"][arr_a];        ts_b = v["temp_enc"][arr_b]

    sleep_diff = _sleep_diff_vec(sl_a, sl_b)

    pairs = pairs.copy()
    pairs["ci_sleep_gap"]   = (sleep_diff >= 2).astype(int)
    pairs["ci_clean_gap"]   = (np.abs(cf_a - cf_b) >= 3).astype(int)
    pairs["ci_snore_noise"] = (((hs_a == 1) & (ns_b >= 4)) | ((hs_b == 1) & (ns_a >= 4))).astype(int)
    pairs["ci_night_call"]  = (np.abs(nc_a - nc_b) >= 2).astype(int)
    pairs["ci_tolerance"]   = (np.abs(td_a - td_b) >= 2).astype(int)
    pairs["ci_smoking"]     = (((sm_a == 0) & (sm_b > 0)) | ((sm_b == 0) & (sm_a > 0))).astype(int)
    pairs["ci_privacy_gap"] = (np.abs(pi_a - pi_b) >= 3).astype(int)
    pairs["ci_eating"]      = (((ei_a <= 2) & (ei_b >= 4)) | ((ei_b <= 2) & (ei_a >= 4))).astype(int)
    pairs["ci_temp"]        = (ts_a != ts_b).astype(int)
    return pairs


def compute_labels(pairs: pd.DataFrame) -> pd.DataFrame:
    """conflict_score → compatible 레이블 생성"""
    weights = np.array([CI_WEIGHTS[c] for c in CI_COLS])
    pairs = pairs.copy()
    pairs["conflict_score"] = pairs[CI_COLS].values @ weights / weights.sum()
    pairs["compatible"]     = (pairs["conflict_score"] < COMPAT_THRESHOLD).astype(int)
    return pairs


def run_feature_engineering(proc_path=None, save=True) -> pd.DataFrame:
    """전체 피처 엔지니어링 파이프라인 실행"""
    print("[feature_engineering] 시작...")
    p = proc_path or PROC_PATH
    df = pd.read_csv(p, index_col="idx")

    # 필수 컬럼 확인
    missing = [c for c in DIFF_COLS + MATCH_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"preprocess.py를 먼저 실행하세요. 없는 컬럼: {missing}")

    v = {col: df[col].values for col in df.columns}

    pairs = make_pairs(df)
    pairs = compute_ci(pairs, v)
    pairs = compute_labels(pairs)

    compat_rate = pairs["compatible"].mean()
    print(f"  compatible=1: {pairs['compatible'].sum():,}  ({compat_rate:.1%})")
    print(f"  compatible=0: {(pairs['compatible']==0).sum():,}  ({1-compat_rate:.1%})")

    # FEATURE_COLS 30개 검증
    missing_feat = [c for c in FEATURE_COLS if c not in pairs.columns]
    if missing_feat:
        raise ValueError(f"피처 누락: {missing_feat}")
    print(f"  피처 검증 OK: {len(FEATURE_COLS)}개")

    if save:
        PAIR_PATH.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(PAIR_PATH, index=False, encoding="utf-8-sig")
        print(f"  저장: {PAIR_PATH}")

    print(f"[feature_engineering] 완료: {len(pairs):,}쌍")
    return pairs


if __name__ == "__main__":
    pairs = run_feature_engineering()
    print(pairs[FEATURE_COLS + ["compatible"]].head())
