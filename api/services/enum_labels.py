# api/services/enum_labels.py
# ==============================================================
# DB ENUM 값 → 한국어 라벨 매핑.
#
# 매칭 리포트에서 사용자에게 보여줄 자연어 설명을 만들 때 사용.
# (예: "BETWEEN_24_2" → "24~2시")
# ==============================================================

BEDTIME_LABELS = {
    "BEFORE_22":      "22시 이전",
    "BETWEEN_22_24":  "22~24시",
    "BETWEEN_24_2":   "24~2시",
    "AFTER_2":        "새벽 2시 이후",
    "IRREGULAR":      "불규칙",
}

WAKE_UP_LABELS = {
    "BEFORE_6":       "6시 이전",
    "BETWEEN_6_8":    "6~8시",
    "BETWEEN_8_10":   "8~10시",
    "AFTER_10":       "10시 이후",
    "IRREGULAR":      "불규칙",
}

CALL_HABIT_LABELS = {
    "WHISPER":        "소곤소곤 조용히",
    "OUTSIDE_ONLY":   "밖에서만",
    "INSIDE_OK":      "방에서도 자유롭게",
}

CLEANING_LABELS = {
    "RARELY":               "거의 안 함",
    "SOMETIMES":            "가끔",
    "ONCE_OR_TWICE_A_WEEK": "주 1~2회",
    "ALMOST_DAILY":         "거의 매일",
}

DORM_STAY_LABELS = {
    "MOSTLY_OUTSIDE":  "주로 밖에서 활동",
    "HALF_AND_HALF":   "절반 정도",
    "MOSTLY_INSIDE":   "주로 방에서 활동",
}

NOISE_LABELS = {
    "VERY_INSENSITIVE":     "매우 둔감",
    "SLIGHTLY_INSENSITIVE": "둔감한 편",
    "NORMAL":               "보통",
    "SLIGHTLY_SENSITIVE":   "민감한 편",
    "VERY_SENSITIVE":       "매우 민감",
}

SMOKING_LABELS = {
    "NON_SMOKER":           "비흡연자",
    "CIGARETTE":            "흡연자",
    "ELECTRONIC_CIGARETTE": "전자담배 사용자",
}

SLEEP_HABIT_LABELS = {
    "NONE":             "없음",
    "TOSS_AND_TURN":    "뒤척임",
    "FREQUENT_WAKING":  "자다 깸",
    "SNORING":          "코골이",
    "TEETH_GRINDING":   "이갈이",
}
