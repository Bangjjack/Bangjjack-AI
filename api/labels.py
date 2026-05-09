# api/labels.py
# ==============================================================
# 피처 키 → 한국어 라벨 매핑
#
# /match-detail 응답의 matchedFeatures, topInfluentialFeatures에
# 들어가는 30개 피처 키를 사용자 화면용 한국어 라벨로 변환할 때 사용.
#
# 사용 예시 (백엔드):
#     from api.labels import to_labels, get_label
#
#     ai_resp = requests.post(".../match-detail", json=...).json()
#     return {
#         "matchRate":          ai_resp["matchRate"],
#         "matchedAttributes":  to_labels(ai_resp["matchedFeatures"]),
#         "recommendedTopics":  to_labels(ai_resp["topInfluentialFeatures"]),
#     }
#
# 라벨 톤 (프론트와 협의 후 자유롭게 수정 가능):
#   - 명사구 형태 ("취침 시간", "잠버릇 (코골이)")
#   - 백엔드/프론트가 문맥에 맞게 감싸서 사용
#       matched 컨텍스트:    "취침 시간이 비슷해요"
#       influential 컨텍스트: "취침 시간에 대해 얘기해 보세요"
# ==============================================================

from typing import Dict, Iterable, List


FEATURE_LABELS: Dict[str, str] = {
    # ── diff_ 그룹 (수치 차이 — 작을수록 잘 맞음) ────────────────
    "diff_sleep_time":         "취침 시간",
    "diff_wake_time":          "기상 시간",
    "diff_night_call":         "밤 통화 빈도",
    "diff_alarm_habit":        "알람 습관",          # DB에 없음 → 기본값
    "diff_clean_freq":         "청소 빈도",
    "diff_eating_in_room":     "방 내 취식 허용도",  # DB에 없음 → 기본값
    "diff_time_at_home":       "기숙사 체류 시간",
    "diff_noise_sensitivity":  "소음 민감도",
    "diff_privacy_importance": "프라이버시 중요도",  # DB에 없음 → 기본값
    "diff_tolerance":          "생활패턴 허용도",     # DB에 없음 → 기본값

    # ── match_ 그룹 (동일 여부) ───────────────────────────────
    # 인적사항 / 흡연
    "match_gender":            "성별",
    "match_smoking":           "흡연 습관",

    # 잠버릇 (5개)
    "match_habit_toss":        "잠버릇 (뒤척임)",
    "match_habit_wakeup":      "잠버릇 (자다 깸)",
    "match_habit_snore":       "잠버릇 (코골이)",
    "match_habit_talk":        "잠버릇 (잠꼬대)",        # DB에 없음 → 기본값
    "match_habit_grind":       "잠버릇 (이갈이)",

    # 기숙사 경험 (3개 — 모두 DB에 없음 → 기본값)
    "match_dorm_exp_none":     "기숙사 경험 (없음)",
    "match_dorm_exp_double":   "기숙사 경험 (2인실)",
    "match_dorm_exp_triple":   "기숙사 경험 (3인실 이상)",

    # 1순위 (5개)
    "match_prio1_sleep":       "1순위: 수면 패턴",
    "match_prio1_clean":       "1순위: 청결도",
    "match_prio1_noise":       "1순위: 소음",
    "match_prio1_smoking":     "1순위: 흡연 여부",
    "match_prio1_rhythm":      "1순위: 생활 리듬",

    # 2순위 (5개)
    "match_prio2_sleep":       "2순위: 수면 패턴",
    "match_prio2_clean":       "2순위: 청결도",
    "match_prio2_noise":       "2순위: 소음",
    "match_prio2_smoking":     "2순위: 흡연 여부",
    "match_prio2_rhythm":      "2순위: 생활 리듬",
}


# ── 헬퍼 함수 ─────────────────────────────────────────────────

def get_label(key: str) -> str:
    """
    단일 피처 키 → 한국어 라벨.
    매핑에 없으면 안전하게 키를 그대로 반환 (서버 무중단).
    """
    return FEATURE_LABELS.get(key, key)


def to_labels(keys: Iterable[str]) -> List[str]:
    """피처 키 리스트 → 한국어 라벨 리스트. 누락 키는 키 그대로 유지."""
    return [get_label(k) for k in keys]
