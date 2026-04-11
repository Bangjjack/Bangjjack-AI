# src/preprocess.py
# ==============================================================
# 1단계: 원본 설문 CSV → bangjjak_processed.csv
# 실행: python src/preprocess.py
# ==============================================================

import sys
from pathlib import Path

# src/ 안에서 config를 import할 수 있도록 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from config import (
    RAW_PATH, PROC_PATH,
    DIFF_COLS, MATCH_COLS,
)


# ── 컬럼 rename 매핑 ──────────────────────────────────────────
COL_MAP = {
    "1. 본인의 성별을 선택해주세요. ":                                               "gender",
    "2. 다인 기숙사 생활 및 공동생활 경험이 있나요? (복수 선택 가능)":                 "shared_living_exp",
    "3. 다인 기숙사 생활 및 공동생활 경험이 있는 경우, 룸메이트와 갈등을 겪어본 적이 있나요?": "had_conflict",
    "4. 과거 공동생활에서 갈등이 있었다면, 주된 원인을 선택해주세요. (복수 선택 가능)":   "conflict_reason",
    "5. 평일 평균 취침 시간이 어떻게 되시나요?":                                       "sleep_time",
    "6. 평일 평균 기상 시간이 어떻게 되시나요?":                                       "wake_time",
    "7. 밤 시간대(22시 이후)에 통화를 얼마나 자주 하나요? ":                            "night_call",
    "8.  본인의 잠버릇에 해당하는 항목을 모두 선택해주세요. (복수 선택 가능) ":          "sleep_habits",
    "9. 본인의 알람 사용 습관에 해당하는 항목을 선택해주세요.":                          "alarm_habit",
    "10. 방 청소를 얼마나 자주 하는 편인가요?":                                        "clean_freq",
    "11. 방 안에서의 취식(배달 음식, 간식 등)에 대해 어떻게 생각하시나요? ":             "eating_in_room",
    "12. 현재 흡연 여부를 선택해주세요. ":                                             "smoking",
    "13. 더위/추위에 대한 민감도를 선택해주세요.  ":                                    "temp_sensitivity",
    "14. 평소 기숙사(또는 집)에서 보내는 시간은 어느 정도인가요?":                       "time_at_home",
    "15. 소음에 얼마나 민감한 편인가요? (1~5점) ":                                     "noise_sensitivity",
    "16.  공동생활에서 프라이버시(개인 공간 및 시간)를 얼마나 중요하게 생각하나요? (1~5점)": "privacy_importance",
    "17. 룸메이트 선택 시 가장 중요하게 생각하는 1순위 조건을 선택해주세요.  ":           "priority_1",
    "18. (선택) 룸메이트 선택 시 그 다음으로 중요하게 생각하는 2순위 조건을 선택해주세요.  ": "priority_2",
    "19. 룸메이트의 생활 패턴이 본인과 어느 정도 달라도 괜찮다고 생각하시나요?":          "tolerance",
    "20. 다음 두 사람 중 함께 생활하고 싶은 사람을 선택해주세요.  \n\nA. 취침 03시 / 청결도 높음(4점) / 소음 둔감(2점) / 비흡연\nB. 취침 23시 / 청결도 보통(2점) / 소음 예민(4점) / 비흡연  ": "scenario_choice",
    "21. 생활 패턴 정보를 기반으로 한 기숙사 룸메이트 매칭 서비스가 있다면 사용하실 의향이 있나요? ": "service_willingness",
}

# ── 순서형 인코딩 매핑 ────────────────────────────────────────
CLEAN_FREQ_MAP = {
    "거의 안 함": 1, "더러울 때": 2, "월 1~2회": 3,
    "주 1회": 4, "주 2~3회": 5, "주 4회": 6, "거의 매일": 7,
}
NOISE_MAP = {
    "거의 신경 쓰지 않음 (1점)": 1, "신경 쓰지 않음 (2점)": 2,
    "보통 (3점)": 3, "민감함 (4점)": 4, "매우 민감함 (5점)": 5,
}
PRIVACY_MAP = {
    "크게 중요하지 않음 (1점)": 1, "중요하지 않음 (2점)": 2,
    "보통 (3점)": 3, "중요함 (4점)": 4, "매우 중요함 (5점)": 5,
}
TOLERANCE_MAP = {
    "거의 비슷해야 한다": 1, "조금 달라도 괜찮다": 2, "많이 달라도 괜찮다": 3,
}
# TODO: 아래 4개는 실제 설문 응답값 확인 후 수정
NIGHT_CALL_MAP = {
    "거의 안 함": 1, "가끔 (주 1~2회)": 2, "자주 (주 3~4회)": 3, "거의 매일": 4,
}
ALARM_MAP = {
    "알람 없이 기상": 1, "알람 1개": 2, "알람 여러 개": 3, "스누즈를 자주 사용": 4,
}
EATING_MAP = {
    "절대 안 먹음": 1, "가능하면 안 먹음": 2, "보통": 3, "가끔 먹음": 4, "상관없음": 5,
}
TIME_AT_HOME_MAP = {
    "거의 안 있음": 1, "잠만 자러 옴": 2, "보통": 3, "주로 있음": 4, "항상 있음": 5,
}
TEMP_MAP = {"더위에 민감": 1, "추위에 민감": 2, "둘 다 민감": 3, "둔감": 0}


# ── 함수 정의 ─────────────────────────────────────────────────
def _time_range_to_hour(val: str) -> float:
    """'HH:MM ~ HH:MM' 형식 → 중앙 시간(float)"""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(" 이후", "~25:00").replace(" ~", "~")
    parts = s.split("~")
    try:
        sh = int(parts[0].strip().split(":")[0])
        eh = int(parts[1].strip().split(":")[0])
        if eh == 0:
            eh = 24
        if eh < sh:
            eh += 24
        return (sh + eh) / 2
    except Exception:
        return np.nan


def _has_kw(val, kw: str) -> int:
    return 1 if isinstance(val, str) and kw in val else 0


def load_raw(path=None) -> pd.DataFrame:
    p = path or RAW_PATH
    df = pd.read_csv(p)
    pii_cols = ["타임스탬프", "성함/연락처",
                "22. (선택) 이벤트 참여를 희망하시는 경우, [성함/연락처]를 기재해주세요. "]
    df = df.drop(columns=pii_cols, errors="ignore")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COL_MAP)


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sleep_time"]         = df["sleep_time"].apply(_time_range_to_hour)
    df["wake_time"]          = df["wake_time"].apply(_time_range_to_hour)
    df["clean_freq"]         = df["clean_freq"].map(CLEAN_FREQ_MAP)
    df["noise_sensitivity"]  = df["noise_sensitivity"].map(NOISE_MAP)
    df["privacy_importance"] = df["privacy_importance"].map(PRIVACY_MAP)
    df["tolerance"]          = df["tolerance"].map(TOLERANCE_MAP)
    df["night_call"]         = df["night_call"].map(NIGHT_CALL_MAP)
    df["alarm_habit"]        = df["alarm_habit"].map(ALARM_MAP)
    df["eating_in_room"]     = df["eating_in_room"].map(EATING_MAP)
    df["time_at_home"]       = df["time_at_home"].map(TIME_AT_HOME_MAP)
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender"]   = df["gender"].map({"남성": 0, "여성": 1})
    df["smoking"]  = df["smoking"].map({"비흡연": 0, "흡연": 1, "전자담배": 2}).fillna(0).astype(int)
    df["temp_enc"] = df["temp_sensitivity"].map(TEMP_MAP).fillna(0).astype(int)
    return df


def encode_multihot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 잠버릇 (Q8) — TODO: 실제 응답 키워드 확인
    df["habit_snore"]  = df["sleep_habits"].apply(lambda x: _has_kw(x, "코골"))
    df["habit_toss"]   = df["sleep_habits"].apply(lambda x: _has_kw(x, "뒤척"))
    df["habit_wakeup"] = df["sleep_habits"].apply(lambda x: _has_kw(x, "자다가 깸"))
    df["habit_talk"]   = df["sleep_habits"].apply(lambda x: _has_kw(x, "잠꼬대"))
    df["habit_grind"]  = df["sleep_habits"].apply(lambda x: _has_kw(x, "이갈이"))

    # 기숙사 경험 (Q2) — TODO: 실제 응답 키워드 확인
    df["dorm_exp_none"]   = df["shared_living_exp"].apply(lambda x: _has_kw(x, "없음"))
    df["dorm_exp_double"] = df["shared_living_exp"].apply(lambda x: _has_kw(x, "2인실"))
    df["dorm_exp_triple"] = df["shared_living_exp"].apply(
        lambda x: 1 if isinstance(x, str) and ("3인실" in x or "4인실" in x) else 0)

    # 우선순위 (Q17, Q18) — TODO: 실제 응답 키워드 확인
    for prefix, src in [("prio1", "priority_1"), ("prio2", "priority_2")]:
        df[f"{prefix}_sleep"]   = df[src].apply(lambda x: _has_kw(x, "수면"))
        df[f"{prefix}_clean"]   = df[src].apply(lambda x: _has_kw(x, "청결"))
        df[f"{prefix}_noise"]   = df[src].apply(lambda x: _has_kw(x, "소음"))
        df[f"{prefix}_smoking"] = df[src].apply(lambda x: _has_kw(x, "흡연"))
        df[f"{prefix}_rhythm"]  = df[src].apply(lambda x: _has_kw(x, "리듬"))

    return df


def fill_na_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DIFF_COLS:
        if col in df.columns and df[col].isna().any():
            n = df[col].isna().sum()
            print(f"  ⚠️  {col}: NaN {n}개 → 중앙값 대체")
            df[col] = df[col].fillna(df[col].median())
    for col in MATCH_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    required = ["sleep_time", "wake_time", "clean_freq", "noise_sensitivity", "tolerance"]
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    df.index.name = "idx"
    print(f"  필터: {before}명 → {len(df)}명 (제거 {before - len(df)}명)")
    return df


def run_preprocess(raw_path=None, save=True) -> pd.DataFrame:
    """전체 전처리 파이프라인 실행"""
    print("[preprocess] 시작...")
    df = load_raw(raw_path)
    df = rename_columns(df)
    df = encode_ordinal(df)
    df = encode_binary(df)
    df = encode_multihot(df)
    df = fill_na_and_filter(df)
    if save:
        PROC_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROC_PATH, index=True, encoding="utf-8-sig")
        print(f"  저장: {PROC_PATH}")
    print(f"[preprocess] 완료: {len(df)}명, {len(df.columns)}컬럼")
    return df


if __name__ == "__main__":
    df = run_preprocess()
    print(df[["sleep_time", "wake_time", "clean_freq", "gender", "smoking"]].head())
