# api/services/report.py
# ==============================================================
# 매칭 리포트 생성 — 체크리스트 8개 항목별 일치/불일치 평가 +
# 사용자 친화적 자연어 설명/조언 + 종합 코멘트.
#
# 같은 응답이 두 화면에서 다르게 활용됨:
#   - 매칭 리포트 (매칭하기 직전): matched + summaryComment.positive 중심
#   - 공동생활 시작 가이드 (매칭 직후):
#       · 공동생활 체크포인트 → matched
#       · 미리 조율하면 좋은점 → mismatched (description + advice)
#       · 대화 추천 주제 → topInfluentialFeatures + mismatched
# ==============================================================

from typing import Dict, List, Tuple

from api.schemas import RawProfile
from api.services.enum_labels import (
    BEDTIME_LABELS,
    CALL_HABIT_LABELS,
    CLEANING_LABELS,
    DORM_STAY_LABELS,
    NOISE_LABELS,
    SLEEP_HABIT_LABELS,
    SMOKING_LABELS,
    WAKE_UP_LABELS,
)


# ── 응답 dict 표준 ──────────────────────────────────────────
# matched:    { "key", "label", "description" }
# mismatched: { "key", "label", "description", "advice" }


# ── 미스매치 시 추천 대화 주제 (요약 코멘트 생성에 사용) ─────
DISCUSSION_TOPICS = {
    "bedtime":           "취침 시간 조정",
    "wake_up_time":      "이른 시간 알람 사용",
    "clean_freq":        "청소 담당과 청소 방식",
    "sleep_habits":      "수면 환경",
    "call_habit":        "통화 가능 시간대",
    "smoking":           "흡연 정책",
    "time_at_home":      "방에서 보내는 시간",
    "noise_sensitivity": "소음 관련 룰",
}


# ── 항목별 평가 함수 ────────────────────────────────────────
# 각 함수는 (status, detail_dict) 반환. status ∈ {"match", "mismatch"}

def _eval_bedtime(a: RawProfile, b: RawProfile):
    if a.bedtime == b.bedtime:
        lbl = BEDTIME_LABELS.get(a.bedtime, a.bedtime)
        return ("match", {
            "key":         "bedtime",
            "label":       "취침 시간",
            "description": f"둘 다 {lbl}로 자는 편이에요",
        })
    return ("mismatch", {
        "key":         "bedtime",
        "label":       "취침 시간",
        "description": f"{BEDTIME_LABELS.get(a.bedtime)}와 "
                       f"{BEDTIME_LABELS.get(b.bedtime)}로 차이가 있어요",
        "advice":      "늦은 시간 조명/소음 룰을 미리 정해두면 좋아요",
    })


def _eval_wake_up_time(a, b):
    if a.wake_up_time == b.wake_up_time:
        lbl = WAKE_UP_LABELS.get(a.wake_up_time)
        return ("match", {
            "key":         "wake_up_time",
            "label":       "기상 시간",
            "description": f"둘 다 {lbl}에 일어나는 편이에요",
        })
    return ("mismatch", {
        "key":         "wake_up_time",
        "label":       "기상 시간",
        "description": f"{WAKE_UP_LABELS.get(a.wake_up_time)}와 "
                       f"{WAKE_UP_LABELS.get(b.wake_up_time)}로 차이가 있어요",
        "advice":      "이른 시간 알람 사용에 대한 룰을 정해두면 좋아요",
    })


def _eval_call_habit(a, b):
    if a.call_habit == b.call_habit:
        lbl = CALL_HABIT_LABELS.get(a.call_habit)
        return ("match", {
            "key":         "call_habit",
            "label":       "통화 습관",
            "description": f"{lbl} 통화하는 것을 선호해요",
        })
    return ("mismatch", {
        "key":         "call_habit",
        "label":       "통화 습관",
        "description": f"통화 습관이 달라요 "
                       f"({CALL_HABIT_LABELS.get(a.call_habit)} ↔ "
                       f"{CALL_HABIT_LABELS.get(b.call_habit)})",
        "advice":      "방 안에서 통화 가능한 시간대를 정해두면 좋아요",
    })


def _eval_cleaning(a, b):
    if a.cleaning_cycle == b.cleaning_cycle:
        lbl = CLEANING_LABELS.get(a.cleaning_cycle)
        return ("match", {
            "key":         "clean_freq",
            "label":       "청소 주기",
            "description": f"둘 다 청소를 {lbl} 하는 편이에요",
        })
    return ("mismatch", {
        "key":         "clean_freq",
        "label":       "청소 주기",
        "description": f"청소 주기가 달라요 "
                       f"({CLEANING_LABELS.get(a.cleaning_cycle)} ↔ "
                       f"{CLEANING_LABELS.get(b.cleaning_cycle)})",
        "advice":      "청소 담당과 청소 방식에 대해 미리 이야기 나눠보세요",
    })


def _eval_dorm_stay(a, b):
    if a.dorm_stay_time == b.dorm_stay_time:
        lbl = DORM_STAY_LABELS.get(a.dorm_stay_time)
        return ("match", {
            "key":         "time_at_home",
            "label":       "기숙사 체류 시간",
            "description": f"둘 다 {lbl} 하는 편이에요",
        })
    return ("mismatch", {
        "key":         "time_at_home",
        "label":       "기숙사 체류 시간",
        "description": "방에서 보내는 시간 패턴이 달라요",
        "advice":      "혼자만의 시간이 필요한 시간대를 서로 알려주세요",
    })


def _eval_noise(a, b):
    if a.noise_sensitivity == b.noise_sensitivity:
        lbl = NOISE_LABELS.get(a.noise_sensitivity)
        return ("match", {
            "key":         "noise_sensitivity",
            "label":       "소음 민감도",
            "description": f"둘 다 소음에 {lbl}한 편이에요",
        })
    return ("mismatch", {
        "key":         "noise_sensitivity",
        "label":       "소음 민감도",
        "description": "소음에 대한 민감도가 달라요",
        "advice":      "이어폰 사용이나 소음 룰을 미리 정해두면 좋아요",
    })


def _eval_smoking(a, b):
    if a.smoking == b.smoking:
        lbl = SMOKING_LABELS.get(a.smoking)
        return ("match", {
            "key":         "smoking",
            "label":       "흡연 여부",
            "description": f"둘 다 {lbl}예요",
        })
    return ("mismatch", {
        "key":         "smoking",
        "label":       "흡연 여부",
        "description": f"흡연 습관이 달라요 "
                       f"({SMOKING_LABELS.get(a.smoking)} ↔ "
                       f"{SMOKING_LABELS.get(b.smoking)})",
        "advice":      "방 내 흡연 정책을 매칭 전 반드시 합의하세요",
    })


def _eval_sleep_habits(a, b):
    a_set = set(a.sleep_habits) - {"NONE"}
    b_set = set(b.sleep_habits) - {"NONE"}
    if a_set == b_set:
        if not a_set:
            return ("match", {
                "key":         "sleep_habits",
                "label":       "잠버릇",
                "description": "둘 다 특별한 잠버릇이 없어요",
            })
        habits = ", ".join(SLEEP_HABIT_LABELS.get(h, h) for h in sorted(a_set))
        return ("match", {
            "key":         "sleep_habits",
            "label":       "잠버릇",
            "description": f"둘 다 {habits} 같은 잠버릇이 있어요",
        })
    return ("mismatch", {
        "key":         "sleep_habits",
        "label":       "잠버릇",
        "description": "잠버릇에 서로 다른 부분이 있어요",
        "advice":      "수면 환경에 영향을 줄 수 있으니 미리 이야기 나눠보세요",
    })


# ── 체크리스트 평가 메인 ────────────────────────────────────
_EVALUATORS = [
    _eval_bedtime,
    _eval_wake_up_time,
    _eval_call_habit,
    _eval_cleaning,
    _eval_sleep_habits,
    _eval_smoking,
    _eval_dorm_stay,
    _eval_noise,
]


def evaluate_checklist(user_a: RawProfile,
                       user_b: RawProfile) -> Tuple[List[dict], List[dict]]:
    """
    두 사용자의 체크리스트 8개 항목별 일치/불일치 평가.
    Returns (matched_items, mismatched_items)
    """
    matched: List[dict] = []
    mismatched: List[dict] = []
    for evaluator in _EVALUATORS:
        status, detail = evaluator(user_a, user_b)
        (matched if status == "match" else mismatched).append(detail)
    return matched, mismatched


# ── 종합 코멘트 생성 ─────────────────────────────────────────

def _positive_intro(match_rate: int) -> str:
    if match_rate >= 70:
        return "생활 습관이 잘 맞아 큰 갈등 없이 지낼 수 있을 것 같아요!"
    if match_rate >= 50:
        return "비슷한 점이 있어 어느 정도 잘 맞을 수 있어요."
    if match_rate >= 30:
        return "일부 항목에서 잘 맞는 점이 있어요."
    return "차이가 큰 편이지만, 일부 일치하는 점은 있어요."


def generate_summary_comment(matched: List[dict],
                             mismatched: List[dict],
                             match_rate: int) -> Dict[str, str]:
    """
    매칭 점수와 항목별 결과 기반으로 자연어 요약 두 문단 생성.
    Returns: { "positive": str, "caution": str }
      - positive: 잘 맞는 점 강조
      - caution:  미리 조율할 점 (없으면 빈 문자열)
    """
    # ── positive 문단 ────────────────────────────────────────
    positive = _positive_intro(match_rate)
    if matched:
        labels = ", ".join(m["label"] for m in matched[:3])
        positive += f" {labels} 항목에서 높은 일치율을 보였어요."

    # ── caution 문단 ─────────────────────────────────────────
    caution = ""
    if mismatched:
        labels = ", ".join(m["label"] for m in mismatched[:2])
        topics = [DISCUSSION_TOPICS.get(m["key"]) for m in mismatched[:2]]
        topics_str = ", ".join(t for t in topics if t)
        if topics_str:
            caution = (
                f"다만 {labels} 항목에서 차이가 있어요. "
                f"룸메이트 매칭 전 {topics_str}에 대해 미리 이야기를 "
                f"나눠보면 더 좋은 룸메이트가 될 수 있을 거예요!"
            )
        else:
            caution = f"다만 {labels} 항목에서 차이가 있어요."

    return {"positive": positive, "caution": caution}
