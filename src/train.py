# src/train.py
# ==============================================================
# 3단계: bangjjak_pairs_v2.csv → LightGBM 모델 학습 + 저장
# 실행: python src/train.py
# ==============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from config import (
    PAIR_PATH, MODEL_PATH,
    FEATURE_COLS, LGB_PARAMS, CV_FOLDS,
)


def train_cv(X: pd.DataFrame, y: pd.Series) -> tuple:
    """5-Fold Stratified CV 학습 → (fold별 모델 리스트, 평균 metrics dict)"""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    auc_list, f1_list, acc_list = [], [], []

    print(f"  [{CV_FOLDS}-Fold CV 시작]  샘플: {len(X):,}  피처: {X.shape[1]}개")
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr])
        prob = m.predict_proba(X.iloc[va])[:, 1]
        pred = (prob >= 0.5).astype(int)
        auc_list.append(roc_auc_score(y.iloc[va], prob))
        f1_list.append(f1_score(y.iloc[va], pred))
        acc_list.append(accuracy_score(y.iloc[va], pred))
        print(f"  Fold {fold}  AUC={auc_list[-1]:.4f}  F1={f1_list[-1]:.4f}  Acc={acc_list[-1]:.4f}")

    metrics = {
        "val_auc": round(float(np.mean(auc_list)), 4),
        "val_f1":  round(float(np.mean(f1_list)),  4),
        "val_acc": round(float(np.mean(acc_list)), 4),
    }
    print(f"\n  ── 평균 ──  AUC: {metrics['val_auc']}  "
          f"F1: {metrics['val_f1']}  Acc: {metrics['val_acc']}")
    return metrics


def train_final(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """전체 데이터로 최종 모델 학습"""
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y)
    train_prob = model.predict_proba(X)[:, 1]
    train_auc  = roc_auc_score(y, train_prob)
    print(f"  Train AUC: {train_auc:.4f}")
    return model, round(train_auc, 4)


def save_model(model, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":         model,
        "feature_names": FEATURE_COLS,   # 30개 이름 — FastAPI 로드 시 사용
        "n_features":    len(FEATURE_COLS),
        "metrics":       metrics,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"  저장: {MODEL_PATH}")
    print(f"  메트릭: {metrics}")


def run_train(pair_path=None) -> dict:
    """전체 학습 파이프라인 실행"""
    print("[train] 시작...")
    p = pair_path or PAIR_PATH
    pairs = pd.read_csv(p)

    # 피처 검증
    missing = [c for c in FEATURE_COLS if c not in pairs.columns]
    if missing:
        raise ValueError(f"feature_engineering.py를 먼저 실행하세요. 없는 피처: {missing}")

    X = pairs[FEATURE_COLS]
    y = pairs["compatible"]
    # ── 여기에 추가 ──────────────────────────────
    print("\n  [타겟 분포 확인]")
    print(y.value_counts())
    print(y.value_counts(normalize=True).round(3))
    # ────────────────────────────────────────────

    assert X.shape[1] == 30, f"피처 수 오류: {X.shape[1]}개 (30개여야 함)"

    # CV 학습
    cv_metrics = train_cv(X, y)

    # 최종 모델 학습
    print("\n  [최종 모델 학습]")
    final_model, train_auc = train_final(X, y)

    val_auc = cv_metrics["val_auc"]
    print(f"  Val  AUC: {val_auc:.4f}")
    print(f"  차이:     {train_auc - val_auc:.4f}  (0.02 이하 권장)")

    # 저장
    metrics = {**cv_metrics, "train_auc": train_auc}
    save_model(final_model, metrics)

    print("[train] 완료")
    return metrics


if __name__ == "__main__":
    metrics = run_train()
    print(f"\n최종 메트릭: {metrics}")
    
