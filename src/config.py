# src/config.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))  # src/ 경로 추가

# ── 프로젝트 루트 경로 ─────────────────────────────────────────
ROOT = Path(__file__).parent.parent   # Bangjjak-AI/

RAW_PATH   = ROOT / "data" / "raw"  / "survey_raw.csv"
PROC_PATH  = ROOT / "data" / "processed" / "bangjjak_processed.csv"
PAIR_PATH  = ROOT / "data" / "processed" / "bangjjak_pairs_v2.csv"
MODEL_PATH = ROOT / "model" / "bangjjak_lgbm_v2.pkl"

# ── 피처 정의 ─────────────────────────────────────────────────
DIFF_COLS = [
    "sleep_time",          # 취침 시간 (시간 단위 변환값)
    "wake_time",           # 기상 시간
    "night_call",          # 밤 통화 빈도
    "alarm_habit",         # 알람 습관
    "clean_freq",          # 청소 빈도
    "eating_in_room",      # 방 내 취식 허용도
    "time_at_home",        # 기숙사 체류 시간
    "noise_sensitivity",   # 소음 민감도
    "privacy_importance",  # 프라이버시 중요도
    "tolerance",           # 생활 패턴 허용 범위
]  # 10개

MATCH_COLS = [
    "gender",
    "smoking",
    "habit_toss",
    "habit_wakeup",
    "habit_snore",
    "habit_talk",
    "habit_grind",
    "dorm_exp_none",
    "dorm_exp_double",
    "dorm_exp_triple",
    "prio1_sleep",
    "prio1_clean",
    "prio1_noise",
    "prio1_smoking",
    "prio1_rhythm",
    "prio2_sleep",
    "prio2_clean",
    "prio2_noise",
    "prio2_smoking",
    "prio2_rhythm",
]  # 20개

FEATURE_COLS = [f"diff_{c}" for c in DIFF_COLS] + \
               [f"match_{c}" for c in MATCH_COLS]   # 30개

CI_COLS = [
    "ci_sleep_gap",
    "ci_clean_gap",
    "ci_snore_noise",
    "ci_night_call",
    "ci_tolerance",
    "ci_smoking",
    "ci_privacy_gap",
    "ci_eating",
    "ci_temp",
]  # 9개 — 레이블 생성 전용

CI_WEIGHTS = {
    "ci_sleep_gap":   0.327,
    "ci_clean_gap":   0.370,
    "ci_snore_noise": 0.279,
    "ci_night_call":  0.140,
    "ci_tolerance":   0.264,
    "ci_smoking":     0.091,
    "ci_privacy_gap": 0.096,
    "ci_eating":      0.020,
    "ci_temp":        0.020,
}

# ── 모델 하이퍼파라미터 ────────────────────────────────────────
LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
)

CV_FOLDS = 5

# ── 검증 목표 ─────────────────────────────────────────────────
SCENARIO_TARGET = 0.65   # 시나리오 일치율 목표 ≥ 65%
COMPAT_THRESHOLD = 0.35   # conflict_score 임계값


if __name__ == "__main__":
    print(f"FEATURE_COLS ({len(FEATURE_COLS)}개): {FEATURE_COLS}")
    print(f"CI_COLS ({len(CI_COLS)}개): {CI_COLS}")
    print(f"ROOT: {ROOT}")
