# api/app.py
# ==============================================================
# FastAPI 진입점 — 모델 로드 + 라우터 등록
# 실행 (프로젝트 루트에서):
#   uvicorn api.app:app --reload
# ==============================================================

import sys
from pathlib import Path

# src/ 경로를 sys.path에 추가 (config.py import용)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import shap
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from config import MODEL_PATH, FEATURE_COLS

# 라우터 import
from api.routes.predict import router as predict_router
from api.routes.explain import router as explain_router
from api.routes.matches import router as matches_router


# ── 서버 시작/종료 시 실행 ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 시작: 모델 로드 ──────────────────────────────────────
    print("[startup] 모델 로드 중...")
    bundle = joblib.load(MODEL_PATH)

    app.state.model        = bundle["model"]
    app.state.feature_names = bundle["feature_names"]   # 30개
    app.state.explainer    = shap.TreeExplainer(bundle["model"])
    app.state.metrics      = bundle.get("metrics", {})

    assert len(app.state.feature_names) == 30, "피처 수 오류"
    print(f"[startup] 완료 — 피처: {len(app.state.feature_names)}개")
    print(f"[startup] 메트릭: {app.state.metrics}")

    yield  # 서버 실행 중

    # ── 종료 ────────────────────────────────────────────────
    print("[shutdown] 서버 종료")


# ── FastAPI 앱 생성 ───────────────────────────────────────────
app = FastAPI(
    title="방짝(BangJjak) AI 매칭 API",
    description="""
## 방짝 — AI 기숙사 룸메이트 매칭 서비스

LightGBM + SHAP 기반 룸메이트 호환성 예측 API.

### 엔드포인트
| 경로 | 설명 |
|------|------|
| `GET /health` | 서버 상태 및 모델 정보 |
| `POST /predict` | 두 사람의 호환 확률 예측 |
| `POST /explain` | SHAP 기반 호환 판단 근거 |
| `POST /top-matches` | 후보자 중 상위 K명 추천 |

### 학습 피처
- **diff_** × 10 : 순서형 차이 (취침시간, 청소빈도 등)
- **match_** × 20 : 범주형 일치 여부 (성별, 흡연, 잠버릇 등)
    """,
    version="2.0",
    lifespan=lifespan,
)


# ── 라우터 등록 ──────────────────────────────────────────────
app.include_router(predict_router, tags=["Prediction"])
app.include_router(explain_router, tags=["Explanation"])
app.include_router(matches_router, tags=["Matching"])


# ── GET /health ───────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="서버 상태 확인")
def health():
    return {
        "status":         "healthy",
        "model_version":  "v2.0",
        "n_features":     len(FEATURE_COLS),
        "metrics":        app.state.metrics,
    }


# ── GET / ─────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"message": "방짝 BangJjak API", "docs": "/docs"}


# ── 직접 실행 ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
