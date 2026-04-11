# src/evaluate.py
# ==============================================================
# 4단계: 성능 평가 (ROC, SHAP) + 5단계: 외부 검증 (시나리오 일치율)
# 실행: python src/evaluate.py
# ==============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")   # 서버/터미널 환경에서 창 없이 저장
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from config import (
    PAIR_PATH, PROC_PATH, MODEL_PATH,
    FEATURE_COLS, SCENARIO_TARGET,
    ROOT,
)

FIG_DIR = ROOT / "model"   # 그래프 저장 위치


def plot_roc(model, X: pd.DataFrame, y: pd.Series, save=True) -> None:
    """ROC Curve 그리기"""
    prob = model.predict_proba(X)[:, 1]
    auc  = roc_auc_score(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (전체 데이터)")
    plt.legend()
    plt.tight_layout()
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = FIG_DIR / "roc_curve.png"
        plt.savefig(path, dpi=150)
        print(f"  저장: {path}")
    plt.show()
    plt.close()
    return auc


def plot_shap(model, X: pd.DataFrame, n_sample=300, save=True) -> None:
    """SHAP 피처 중요도 (bar plot)"""
    sample_X = X.sample(min(n_sample, len(X)), random_state=42)
    exp      = shap.TreeExplainer(model)
    sv       = exp.shap_values(sample_X)

    plt.figure()
    shap.summary_plot(sv, sample_X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        path = FIG_DIR / "shap_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  저장: {path}")
    plt.show()
    plt.close()


def validate_scenario(model, pairs: pd.DataFrame, df: pd.DataFrame) -> float:
    """시나리오 A/B 일치율 외부 검증"""
    # scenario_choice 인코딩
    sc_map = {"A": 0, "B": 1}
    df_sc = df.copy()
    df_sc["scenario_enc"] = df_sc["scenario_choice"].map(sc_map)
    valid_idx = set(df_sc.dropna(subset=["scenario_enc"]).index)

    # 두 사람 모두 시나리오 응답이 있는 쌍만
    mask = pairs["idx_a"].isin(valid_idx) & pairs["idx_b"].isin(valid_idx)
    sc = pairs[mask].copy().reset_index(drop=True)

    sc_a = df_sc.loc[sc["idx_a"].values, "scenario_enc"].values
    sc_b = df_sc.loc[sc["idx_b"].values, "scenario_enc"].values
    sc["scenario_match"] = (sc_a == sc_b).astype(int)

    pred = model.predict(sc[FEATURE_COLS])
    compat_mask = pred == 1
    match_rate  = sc.loc[compat_mask, "scenario_match"].mean()

    print(f"  검증 쌍 수:         {len(sc):,}")
    print(f"  compatible=1 예측:  {compat_mask.sum():,}쌍")
    print(f"  시나리오 일치율:    {match_rate:.1%}  (목표 ≥ {SCENARIO_TARGET:.0%})")
    status = "✅ 목표 달성" if match_rate >= SCENARIO_TARGET else "⚠️  목표 미달 (임계값/피처 조정 검토)"
    print(f"  {status}")
    return float(match_rate)


def run_evaluate(pair_path=None, proc_path=None, model_path=None) -> dict:
    """전체 평가 파이프라인 실행"""
    print("[evaluate] 시작...")

    # 데이터 로드
    pairs = pd.read_csv(pair_path or PAIR_PATH)
    df    = pd.read_csv(proc_path or PROC_PATH, index_col="idx")

    # 모델 로드
    bundle = joblib.load(model_path or MODEL_PATH)
    model  = bundle["model"]

    X = pairs[FEATURE_COLS]
    y = pairs["compatible"]

    # 4단계: ROC + SHAP
    print("\n  [4단계: 성능 평가]")
    train_auc = plot_roc(model, X, y)
    plot_shap(model, X)

    # 5단계: 외부 검증
    print("\n  [5단계: 외부 검증]")
    match_rate = validate_scenario(model, pairs, df)

    # 메트릭 업데이트 후 모델 재저장
    bundle["metrics"]["train_auc"]           = round(train_auc, 4)
    bundle["metrics"]["scenario_match_rate"] = round(match_rate, 4)
    joblib.dump(bundle, model_path or MODEL_PATH)
    print(f"\n  메트릭 업데이트 완료: {bundle['metrics']}")

    print("[evaluate] 완료")
    return bundle["metrics"]


if __name__ == "__main__":
    metrics = run_evaluate()
    print(f"\n최종 메트릭: {metrics}")
