# api/schemas.py
# ==============================================================
# Pydantic 요청/응답 모델 정의
# ==============================================================

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ── 입력: 한 사람의 프로필 ────────────────────────────────────
class UserProfile(BaseModel):
    # ── diff_ 계산용 (순서형 — 절댓값 차이) ──────────────────
    sleep_time:          float = Field(..., description="취침 시간 (예: 23.5 = 23:30)")
    wake_time:           float = Field(..., description="기상 시간 (예: 7.5 = 07:30)")
    night_call:          int   = Field(..., ge=1, le=4, description="밤 통화 빈도 1=거의안함 ~ 4=거의매일")
    alarm_habit:         int   = Field(..., ge=1, le=4, description="알람 습관 1=없이기상 ~ 4=스누즈자주")
    clean_freq:          int   = Field(..., ge=1, le=7, description="청소 빈도 1=거의안함 ~ 7=거의매일")
    eating_in_room:      int   = Field(..., ge=1, le=5, description="방 내 취식 허용도 1=절대안먹음 ~ 5=상관없음")
    time_at_home:        int   = Field(..., ge=1, le=5, description="기숙사 체류 시간 1=거의안있음 ~ 5=항상있음")
    noise_sensitivity:   int   = Field(..., ge=1, le=5, description="소음 민감도 1=둔감 ~ 5=매우민감")
    privacy_importance:  int   = Field(..., ge=1, le=5, description="프라이버시 중요도 1=중요안함 ~ 5=매우중요")
    tolerance:           int   = Field(..., ge=1, le=3, description="생활패턴 허용 1=비슷해야함 2=조금달라도됨 3=많이달라도됨")

    # ── match_ 계산용 (범주형/이진형 — 동일 여부) ────────────
    gender:              int   = Field(..., ge=0, le=1,  description="성별 0=남 1=여")
    smoking:             int   = Field(..., ge=0, le=2,  description="흡연 0=비흡연 1=흡연 2=전자담배")
    habit_toss:          int   = Field(..., ge=0, le=1,  description="잠버릇: 뒤척임")
    habit_wakeup:        int   = Field(..., ge=0, le=1,  description="잠버릇: 자다가 깸")
    habit_snore:         int   = Field(..., ge=0, le=1,  description="잠버릇: 코골이")
    habit_talk:          int   = Field(..., ge=0, le=1,  description="잠버릇: 잠꼬대")
    habit_grind:         int   = Field(..., ge=0, le=1,  description="잠버릇: 이갈이")
    dorm_exp_none:       int   = Field(..., ge=0, le=1,  description="기숙사 경험: 없음")
    dorm_exp_double:     int   = Field(..., ge=0, le=1,  description="기숙사 경험: 2인실")
    dorm_exp_triple:     int   = Field(..., ge=0, le=1,  description="기숙사 경험: 3인실 이상")
    prio1_sleep:         int   = Field(..., ge=0, le=1,  description="1순위: 수면 패턴")
    prio1_clean:         int   = Field(..., ge=0, le=1,  description="1순위: 청결도")
    prio1_noise:         int   = Field(..., ge=0, le=1,  description="1순위: 소음")
    prio1_smoking:       int   = Field(..., ge=0, le=1,  description="1순위: 흡연 여부")
    prio1_rhythm:        int   = Field(..., ge=0, le=1,  description="1순위: 생활 리듬")
    prio2_sleep:         int   = Field(..., ge=0, le=1,  description="2순위: 수면 패턴")
    prio2_clean:         int   = Field(..., ge=0, le=1,  description="2순위: 청결도")
    prio2_noise:         int   = Field(..., ge=0, le=1,  description="2순위: 소음")
    prio2_smoking:       int   = Field(..., ge=0, le=1,  description="2순위: 흡연 여부")
    prio2_rhythm:        int   = Field(..., ge=0, le=1,  description="2순위: 생활 리듬")

    class Config:
        json_schema_extra = {
            "example": {
                "sleep_time": 24.0, "wake_time": 7.5,
                "night_call": 1, "alarm_habit": 2,
                "clean_freq": 4, "eating_in_room": 3,
                "time_at_home": 3, "noise_sensitivity": 3,
                "privacy_importance": 3, "tolerance": 2,
                "gender": 1, "smoking": 0,
                "habit_toss": 0, "habit_wakeup": 0, "habit_snore": 0,
                "habit_talk": 0, "habit_grind": 0,
                "dorm_exp_none": 0, "dorm_exp_double": 1, "dorm_exp_triple": 0,
                "prio1_sleep": 1, "prio1_clean": 0, "prio1_noise": 0,
                "prio1_smoking": 0, "prio1_rhythm": 0,
                "prio2_sleep": 0, "prio2_clean": 1, "prio2_noise": 0,
                "prio2_smoking": 0, "prio2_rhythm": 0,
            }
        }


# ── /predict 요청/응답 ────────────────────────────────────────
class PairRequest(BaseModel):
    user_a: UserProfile
    user_b: UserProfile


class PredictResponse(BaseModel):
    compatible:  int   = Field(..., description="호환 여부 (1=호환, 0=비호환)")
    probability: float = Field(..., description="호환 확률 (0.0 ~ 1.0)")


# ── /explain 요청/응답 ────────────────────────────────────────
class FeatureContribution(BaseModel):
    feature:    str   = Field(..., description="피처 이름")
    shap_value: float = Field(..., description="SHAP 기여도 (양수=호환에 유리, 음수=비호환에 유리)")


class ExplainResponse(BaseModel):
    compatible:   int
    probability:  float
    top_features: List[FeatureContribution] = Field(..., description="영향력 상위 5개 피처")


# ── /top-matches 요청/응답 ────────────────────────────────────
class TopMatchRequest(BaseModel):
    user:       UserProfile
    candidates: List[UserProfile] = Field(..., description="비교할 후보자 목록")
    top_k:      int = Field(default=5, ge=1, le=50, description="반환할 상위 매칭 수")


class MatchResult(BaseModel):
    rank:            int
    candidate_index: int
    probability:     float


class TopMatchResponse(BaseModel):
    top_matches: List[MatchResult]


# ============================================================
# /match-detail 전용 스키마 — DB ENUM 그대로 받는 입력
# ============================================================

# ── DB ENUM 타입 정의 (Literal로 검증) ───────────────────────
GenderEnum     = Literal["MALE", "FEMALE"]
BedtimeEnum    = Literal["BEFORE_22", "BETWEEN_22_24", "BETWEEN_24_2",
                         "AFTER_2", "IRREGULAR"]
WakeUpEnum     = Literal["BEFORE_6", "BETWEEN_6_8", "BETWEEN_8_10",
                         "AFTER_10", "IRREGULAR"]
CallHabitEnum  = Literal["WHISPER", "OUTSIDE_ONLY", "INSIDE_OK"]
CleaningEnum   = Literal["RARELY", "SOMETIMES", "ONCE_OR_TWICE_A_WEEK",
                         "ALMOST_DAILY"]
DormStayEnum   = Literal["MOSTLY_OUTSIDE", "HALF_AND_HALF", "MOSTLY_INSIDE"]
NoiseEnum      = Literal["VERY_INSENSITIVE", "SLIGHTLY_INSENSITIVE",
                         "NORMAL", "SLIGHTLY_SENSITIVE", "VERY_SENSITIVE"]
SmokingEnum    = Literal["NON_SMOKER", "CIGARETTE", "ELECTRONIC_CIGARETTE"]
SleepHabitEnum = Literal["NONE", "TOSS_AND_TURN", "FREQUENT_WAKING",
                         "SNORING", "TEETH_GRINDING"]
PriorityEnum   = Literal["BEDTIME", "CALL_HABIT", "CLEANING_HABIT",
                         "DORM_STAY_TIME", "INDOOR_TEMPERATURE",
                         "ITEM_SHARING", "NOISE_SENSITIVITY",
                         "SLEEP_HABIT", "SMOKING", "WAKE_UP_TIME"]


# ── DB ENUM을 그대로 받는 입력 스키마 ────────────────────────
class RawProfile(BaseModel):
    """백엔드가 DB ENUM 그대로 보내는 입력 형식."""
    gender:            GenderEnum            = Field(..., description="users.gender")
    bedtime:           BedtimeEnum           = Field(..., description="lifestyle_checklists.bedtime")
    wake_up_time:      WakeUpEnum            = Field(..., description="lifestyle_checklists.wake_up_time")
    call_habit:        CallHabitEnum         = Field(..., description="lifestyle_checklists.call_habit")
    cleaning_cycle:    CleaningEnum          = Field(..., description="lifestyle_checklists.cleaning_cycle")
    dorm_stay_time:    DormStayEnum          = Field(..., description="lifestyle_checklists.dorm_stay_time")
    noise_sensitivity: NoiseEnum             = Field(..., description="lifestyle_checklists.noise_sensitivity")
    smoking:           SmokingEnum           = Field(..., description="lifestyle_checklists.smoking")
    sleep_habits:      List[SleepHabitEnum]  = Field(default_factory=list,
                                                     description="lifestyle_checklist_sleep_habits — 다중 선택")
    first_priority:    PriorityEnum          = Field(..., description="roommate_preferences.first_priority")
    second_priority:   PriorityEnum          = Field(..., description="roommate_preferences.second_priority")
    third_priority:    PriorityEnum          = Field(..., description="roommate_preferences.third_priority (현재 모델은 미사용)")

    class Config:
        json_schema_extra = {
            "example": {
                "gender":            "FEMALE",
                "bedtime":           "BETWEEN_22_24",
                "wake_up_time":      "BETWEEN_6_8",
                "call_habit":        "WHISPER",
                "cleaning_cycle":    "ONCE_OR_TWICE_A_WEEK",
                "dorm_stay_time":    "MOSTLY_INSIDE",
                "noise_sensitivity": "NORMAL",
                "smoking":           "NON_SMOKER",
                "sleep_habits":      ["TOSS_AND_TURN", "FREQUENT_WAKING"],
                "first_priority":    "CLEANING_HABIT",
                "second_priority":   "NOISE_SENSITIVITY",
                "third_priority":    "SLEEP_HABIT",
            }
        }


class MatchDetailRequest(BaseModel):
    user_a: RawProfile
    user_b: RawProfile

    class Config:
        # Swagger UI가 "Try it out"을 펼쳤을 때 서로 다른 두 사용자가
        # 기본 예시로 채워지도록 명시. (RawProfile의 단일 example을
        # 양쪽에 그대로 채우면 user_a == user_b 가 되어 matchRate=100
        # 으로 보이기 때문에 명시적으로 두 개의 다른 예시를 지정한다.)
        json_schema_extra = {
            "example": {
                "user_a": {
                    "gender":            "FEMALE",
                    "bedtime":           "BETWEEN_22_24",
                    "wake_up_time":      "BETWEEN_6_8",
                    "call_habit":        "WHISPER",
                    "cleaning_cycle":    "ONCE_OR_TWICE_A_WEEK",
                    "dorm_stay_time":    "MOSTLY_INSIDE",
                    "noise_sensitivity": "NORMAL",
                    "smoking":           "NON_SMOKER",
                    "sleep_habits":      ["TOSS_AND_TURN", "FREQUENT_WAKING"],
                    "first_priority":    "CLEANING_HABIT",
                    "second_priority":   "NOISE_SENSITIVITY",
                    "third_priority":    "SLEEP_HABIT",
                },
                "user_b": {
                    "gender":            "FEMALE",
                    "bedtime":           "AFTER_2",
                    "wake_up_time":      "AFTER_10",
                    "call_habit":        "INSIDE_OK",
                    "cleaning_cycle":    "RARELY",
                    "dorm_stay_time":    "MOSTLY_OUTSIDE",
                    "noise_sensitivity": "VERY_SENSITIVE",
                    "smoking":           "ELECTRONIC_CIGARETTE",
                    "sleep_habits":      ["SNORING"],
                    "first_priority":    "SMOKING",
                    "second_priority":   "SLEEP_HABIT",
                    "third_priority":    "BEDTIME",
                },
            }
        }


class MatchDetailResponse(BaseModel):
    matchRate: int = Field(
        ..., ge=0, le=100,
        description="매칭률 (0~100 정수, logit 보정 적용 — 보통 10~90 범위)"
    )
    matchedFeatures: List[str] = Field(
        ...,
        description="match_* 피처 중 두 사용자가 일치한 항목 키 목록"
    )
    topInfluentialFeatures: List[str] = Field(
        ...,
        description="SHAP 절댓값 기준 영향력 Top 3 피처 키"
    )


# ============================================================
# /match-detail-batch — 1:N 배치 매칭 스키마
# ============================================================

class MatchDetailBatchRequest(BaseModel):
    user: RawProfile = Field(..., description="기준 사용자")
    candidates: List[RawProfile] = Field(
        ..., min_length=1,
        description="비교 대상 후보자 목록 (최소 1명)"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1,
        description="반환할 상위 추천 수 (생략 시 전체 후보를 정렬해 반환)"
    )

    class Config:
        # Swagger UI 기본 예시 — 1명 기준 + 후보 3명(잘 맞음/안 맞음/중간)
        json_schema_extra = {
            "example": {
                "user": {
                    "gender":            "FEMALE",
                    "bedtime":           "BETWEEN_22_24",
                    "wake_up_time":      "BETWEEN_6_8",
                    "call_habit":        "WHISPER",
                    "cleaning_cycle":    "ONCE_OR_TWICE_A_WEEK",
                    "dorm_stay_time":    "MOSTLY_INSIDE",
                    "noise_sensitivity": "NORMAL",
                    "smoking":           "NON_SMOKER",
                    "sleep_habits":      ["TOSS_AND_TURN"],
                    "first_priority":    "CLEANING_HABIT",
                    "second_priority":   "NOISE_SENSITIVITY",
                    "third_priority":    "SLEEP_HABIT",
                },
                "candidates": [
                    {  # 후보 0: 잘 맞을 사람 (취침만 살짝 다름)
                        "gender":            "FEMALE",
                        "bedtime":           "BETWEEN_24_2",
                        "wake_up_time":      "BETWEEN_6_8",
                        "call_habit":        "WHISPER",
                        "cleaning_cycle":    "ONCE_OR_TWICE_A_WEEK",
                        "dorm_stay_time":    "MOSTLY_INSIDE",
                        "noise_sensitivity": "NORMAL",
                        "smoking":           "NON_SMOKER",
                        "sleep_habits":      [],
                        "first_priority":    "CLEANING_HABIT",
                        "second_priority":   "NOISE_SENSITIVITY",
                        "third_priority":    "SLEEP_HABIT",
                    },
                    {  # 후보 1: 극도 비호환 (흡연 + 생활 정반대)
                        "gender":            "FEMALE",
                        "bedtime":           "AFTER_2",
                        "wake_up_time":      "AFTER_10",
                        "call_habit":        "INSIDE_OK",
                        "cleaning_cycle":    "RARELY",
                        "dorm_stay_time":    "MOSTLY_OUTSIDE",
                        "noise_sensitivity": "VERY_SENSITIVE",
                        "smoking":           "ELECTRONIC_CIGARETTE",
                        "sleep_habits":      ["SNORING"],
                        "first_priority":    "SMOKING",
                        "second_priority":   "SLEEP_HABIT",
                        "third_priority":    "BEDTIME",
                    },
                    {  # 후보 2: 중간 — 몇 가지 다름
                        "gender":            "FEMALE",
                        "bedtime":           "BETWEEN_24_2",
                        "wake_up_time":      "BETWEEN_8_10",
                        "call_habit":        "OUTSIDE_ONLY",
                        "cleaning_cycle":    "SOMETIMES",
                        "dorm_stay_time":    "HALF_AND_HALF",
                        "noise_sensitivity": "SLIGHTLY_SENSITIVE",
                        "smoking":           "NON_SMOKER",
                        "sleep_habits":      ["TOSS_AND_TURN"],
                        "first_priority":    "SLEEP_HABIT",
                        "second_priority":   "CLEANING_HABIT",
                        "third_priority":    "NOISE_SENSITIVITY",
                    },
                ],
                "top_k": 3,
            }
        }


class MatchDetailBatchItem(BaseModel):
    rank: int = Field(..., ge=1, description="순위 (1이 가장 호환됨)")
    candidateIndex: int = Field(
        ..., ge=0,
        description="요청 candidates 배열에서의 원본 인덱스"
    )
    matchRate: int = Field(..., ge=0, le=100, description="매칭률 (logit 보정)")
    matchedFeatures: List[str] = Field(
        ...,
        description="의미 있게 일치한 match_* 피처 키 목록"
    )
    topInfluentialFeatures: List[str] = Field(
        ...,
        description="SHAP 절댓값 기준 영향력 Top 3 피처 키"
    )


class MatchDetailBatchResponse(BaseModel):
    ranked: List[MatchDetailBatchItem] = Field(
        ...,
        description="matchRate 내림차순 정렬 결과 (top_k 적용 후)"
    )
