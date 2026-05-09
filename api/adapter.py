# api/adapter.py
# ==============================================================
# DB ENUM ↔ AI 모델 입력(숫자) 변환 어댑터
#
# 백엔드는 DB ENUM 그대로 보내고,
# 이 모듈이 30개 피처 형식으로 변환합니다.
#
# DB에 없는 피처(alarm_habit, eating_in_room, privacy_importance,
# tolerance, habit_talk, dorm_exp_*)는 중간값(DEFAULTS)으로 채웁니다.
#
# 모든 사용자에게 같은 값이 들어가므로, diff_*는 항상 0이 되어
# 모델이 사실상 해당 피처를 무시하게 됩니다.
# ==============================================================

from typing import Dict
from api.schemas import RawProfile


# ── DIFF 계열 ENUM → 숫자 매핑 ───────────────────────────────

# 취침 시간 (24시간 단위, sleep_diff()는 자정 순환 보정 처리)
BEDTIME_MAP: Dict[str, float] = {
    "BEFORE_22":      21.0,
    "BETWEEN_22_24":  23.0,
    "BETWEEN_24_2":   1.0,
    "AFTER_2":        3.0,
    "IRREGULAR":      24.0,   # 자정 = 0/24시. _sleep_diff가 cyclic으로 처리
}

# 기상 시간 (24시간)
WAKE_UP_MAP: Dict[str, float] = {
    "BEFORE_6":       5.0,
    "BETWEEN_6_8":    7.0,
    "BETWEEN_8_10":   9.0,
    "AFTER_10":       11.0,
    "IRREGULAR":      7.0,    # 평균값
}

# 통화 습관 (AI: 1~4, DB: 3단계 — 1, 2, 4 사용)
CALL_HABIT_MAP: Dict[str, int] = {
    "WHISPER":        1,   # 작게 통화
    "OUTSIDE_ONLY":   2,   # 밖에서만
    "INSIDE_OK":      4,   # 방에서도 OK (잦음)
}

# 청소 주기 (AI: 1~7, DB: 4단계 — 1, 3, 5, 7)
CLEANING_MAP: Dict[str, int] = {
    "RARELY":               1,
    "SOMETIMES":            3,
    "ONCE_OR_TWICE_A_WEEK": 5,
    "ALMOST_DAILY":         7,
}

# 기숙사 체류 시간 (AI: 1~5, DB: 3단계 — 1, 3, 5)
DORM_STAY_MAP: Dict[str, int] = {
    "MOSTLY_OUTSIDE":       1,
    "HALF_AND_HALF":        3,
    "MOSTLY_INSIDE":        5,
}

# 소음 민감도 (AI: 1~5, DB: 5단계 — 1:1 매핑)
NOISE_MAP: Dict[str, int] = {
    "VERY_INSENSITIVE":     1,
    "SLIGHTLY_INSENSITIVE": 2,
    "NORMAL":               3,
    "SLIGHTLY_SENSITIVE":   4,
    "VERY_SENSITIVE":       5,
}


# ── MATCH 계열 ENUM → 숫자 매핑 ──────────────────────────────

GENDER_MAP: Dict[str, int] = {
    "MALE":   0,
    "FEMALE": 1,
}

SMOKING_MAP: Dict[str, int] = {
    "NON_SMOKER":           0,
    "CIGARETTE":            1,
    "ELECTRONIC_CIGARETTE": 2,
}

# DB 잠버릇 ENUM → AI habit_* 필드명
# habit_talk(잠꼬대)는 DB에 없으므로 SLEEP_HABIT_MAP에 없음 → 기본값 0 유지
SLEEP_HABIT_MAP: Dict[str, str] = {
    "TOSS_AND_TURN":   "habit_toss",
    "FREQUENT_WAKING": "habit_wakeup",
    "SNORING":         "habit_snore",
    "TEETH_GRINDING":  "habit_grind",
    # "NONE" 은 의미상 "없음" → 매핑 안 함 (전부 0 유지)
}

# DB 우선순위 카테고리 → AI 모델 카테고리(sleep/clean/noise/smoking/rhythm)
# AI 모델에 없는 카테고리(CALL_HABIT, INDOOR_TEMPERATURE, ITEM_SHARING)는 무시
PRIORITY_MAP: Dict[str, str] = {
    "BEDTIME":           "rhythm",   # 취침 시간 → 생활 리듬
    "WAKE_UP_TIME":      "rhythm",   # 기상 시간 → 생활 리듬
    "DORM_STAY_TIME":    "rhythm",   # 체류 시간 → 생활 리듬
    "SLEEP_HABIT":       "sleep",    # 잠버릇 → 수면 패턴
    "CLEANING_HABIT":    "clean",
    "NOISE_SENSITIVITY": "noise",
    "SMOKING":           "smoking",
    # 매핑 없음: CALL_HABIT, INDOOR_TEMPERATURE, ITEM_SHARING
}


# ── DB에 없는 피처에 대한 기본값(중간값) ─────────────────────
DEFAULTS: Dict[str, int] = {
    "alarm_habit":         2,   # 1~4 중간
    "eating_in_room":      3,   # 1~5 중간
    "privacy_importance":  3,   # 1~5 중간
    "tolerance":           2,   # 1~3 중간
    "habit_talk":          0,   # 잠꼬대 — 없음으로 가정
    "dorm_exp_none":       0,
    "dorm_exp_double":     0,
    "dorm_exp_triple":     0,
}


# DEFAULTS로 채워진 피처들의 build_features 결과 키 이름.
# 이 피처들은 두 사용자가 모두 기본값을 갖기 때문에 항상 "일치"로 잡힘.
# matchedFeatures / topInfluentialFeatures 응답에서 제외해야 무의미한
# 매칭 노이즈를 줄일 수 있음.
DEFAULT_FEATURE_KEYS: set = {
    # diff_ 그룹은 항상 0 → 사실상 무의미 (모델이 무시)
    "diff_alarm_habit",
    "diff_eating_in_room",
    "diff_privacy_importance",
    "diff_tolerance",
    # match_ 그룹은 항상 1(둘 다 0이라서 일치) → 응답에서 제외
    "match_habit_talk",
    "match_dorm_exp_none",
    "match_dorm_exp_double",
    "match_dorm_exp_triple",
}


# one-hot 인코딩된 match_ 피처 키.
#
# 이 피처들은 둘 다 0이어도 "동일"로 잡힘 (예: 둘 다 1순위가 sleep 아니면
# match_prio1_sleep = 1). matchedFeatures에 그대로 노출하면 의미 없는
# false positive(둘 다 그 항목을 안 가짐)가 화면에 일치로 표시되어
# 사용자 혼동을 유발하므로, "둘 다 값이 1"인 경우에만 매칭으로 인정.
#
# match_gender(0/1), match_smoking(0/1/2)은 둘 다 0이어도 의미 있는
# 일치(둘 다 남자 / 둘 다 비흡연)이므로 이 셋에 포함하지 않음.
ONE_HOT_MATCH_KEYS: set = {
    # 잠버릇 (4개 — habit_talk은 DEFAULT로 이미 제외)
    "match_habit_toss",
    "match_habit_wakeup",
    "match_habit_snore",
    "match_habit_grind",
    # 1순위 (5개)
    "match_prio1_sleep",
    "match_prio1_clean",
    "match_prio1_noise",
    "match_prio1_smoking",
    "match_prio1_rhythm",
    # 2순위 (5개)
    "match_prio2_sleep",
    "match_prio2_clean",
    "match_prio2_noise",
    "match_prio2_smoking",
    "match_prio2_rhythm",
}


# ── 메인 변환 함수 ────────────────────────────────────────────
def raw_to_user_profile(raw: RawProfile) -> dict:
    """
    RawProfile (DB ENUM 입력) → AI 내부 dict (UserProfile.dict()와 동일 형식).
    api.features.build_features()에 그대로 전달 가능.
    """
    p: dict = {}

    # ── DIFF 그룹 (10개) ─────────────────────────────────────
    p["sleep_time"]         = BEDTIME_MAP[raw.bedtime]
    p["wake_time"]          = WAKE_UP_MAP[raw.wake_up_time]
    p["night_call"]         = CALL_HABIT_MAP[raw.call_habit]
    p["clean_freq"]         = CLEANING_MAP[raw.cleaning_cycle]
    p["time_at_home"]       = DORM_STAY_MAP[raw.dorm_stay_time]
    p["noise_sensitivity"]  = NOISE_MAP[raw.noise_sensitivity]
    # DB에 없음 → 중간값
    p["alarm_habit"]        = DEFAULTS["alarm_habit"]
    p["eating_in_room"]     = DEFAULTS["eating_in_room"]
    p["privacy_importance"] = DEFAULTS["privacy_importance"]
    p["tolerance"]          = DEFAULTS["tolerance"]

    # ── MATCH 그룹 (20개) ────────────────────────────────────
    # 인적 사항 / 흡연
    p["gender"]   = GENDER_MAP[raw.gender]
    p["smoking"]  = SMOKING_MAP[raw.smoking]

    # 잠버릇 (5개) — 모두 0으로 초기화 후, sleep_habits에 들어있는 항목만 1로
    p["habit_toss"]   = 0
    p["habit_wakeup"] = 0
    p["habit_snore"]  = 0
    p["habit_talk"]   = DEFAULTS["habit_talk"]    # DB에 없음
    p["habit_grind"]  = 0

    for habit_enum in raw.sleep_habits:
        ai_field = SLEEP_HABIT_MAP.get(habit_enum)
        if ai_field is not None:
            p[ai_field] = 1

    # 기숙사 경험 (3개) — DB에 없음
    p["dorm_exp_none"]   = DEFAULTS["dorm_exp_none"]
    p["dorm_exp_double"] = DEFAULTS["dorm_exp_double"]
    p["dorm_exp_triple"] = DEFAULTS["dorm_exp_triple"]

    # 우선순위 (10개) — 1순위/2순위만 사용 (3순위는 모델에 없음)
    # 모두 0으로 초기화 후 매핑되는 카테고리만 1로
    for prio in ("prio1", "prio2"):
        for cat in ("sleep", "clean", "noise", "smoking", "rhythm"):
            p[f"{prio}_{cat}"] = 0

    cat1 = PRIORITY_MAP.get(raw.first_priority)
    if cat1 is not None:
        p[f"prio1_{cat1}"] = 1

    cat2 = PRIORITY_MAP.get(raw.second_priority)
    if cat2 is not None:
        p[f"prio2_{cat2}"] = 1

    # 3순위(raw.third_priority)는 의도적으로 미사용

    return p
