import uvicorn
import pandas as pd
import numpy as np
import pickle
import os
import traceback
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

# --- CONFIG ---
MODEL_FILE = "models_pro.pkl"
LOG_BUFFER = []  # Хранилище логов для отправки на фронт

# --- APP SETUP ---
app = FastAPI(title="Views Predictor Dashboard")

# Разрешаем CORS (на всякий случай)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное хранилище моделей
ARTIFACTS = None


# --- LOGGER ---
def backend_log(message: str, level="INFO"):
    """Пишет лог и в консоль, и в буфер для веб-страницы"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] [{level}] {message}"
    print(full_msg)
    LOG_BUFFER.append({"msg": full_msg, "level": level})
    # Держим буфер не слишком большим
    if len(LOG_BUFFER) > 500:
        LOG_BUFFER.pop(0)


# --- MODEL UTILS ---
def load_artifacts():
    global ARTIFACTS
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                ARTIFACTS = pickle.load(f)
            backend_log(f"Модели успешно загружены из {MODEL_FILE}", "SUCCESS")
            return True
        else:
            backend_log(f"Файл {MODEL_FILE} не найден. Начинаю обучение...", "WARNING")
            return False
    except Exception as e:
        backend_log(f"Ошибка загрузки: {str(e)}", "ERROR")
        return False


def preprocess(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['month'] = df['DATE'].dt.month
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['day'] = df['DATE'].dt.day
    df['is_weekend'] = (df['DATE'].dt.dayofweek >= 5).astype(int)
    if 'CPM' in df.columns:
        df['log_cpm'] = np.log1p(df['CPM'])
    if 'CHANNEL_NAME' in df.columns:
        df['text_channel'] = df['CHANNEL_NAME'].astype(str).fillna("unknown").str.lower()
    return df


# --- HTML INTERFACE ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML System Monitor</title>
        <style>
            body { background-color: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace; padding: 20px; }
            .container { max-width: 900px; margin: 0 auto; }
            h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }

            #console {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 6px;
                height: 400px;
                overflow-y: auto;
                padding: 10px;
                margin-bottom: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.5);
            }
            .log-line { margin: 2px 0; font-size: 14px; }
            .INFO { color: #8b949e; }
            .WARNING { color: #d29922; }
            .ERROR { color: #f85149; font-weight: bold; }
            .SUCCESS { color: #3fb950; font-weight: bold; }

            #control-panel { 
                background: #161b22; 
                padding: 20px; 
                border-radius: 6px; 
                border: 1px solid #30363d;
                display: none; /* Скрыто до конца обучения */
            }
            input, button {
                padding: 10px;
                margin: 5px 0;
                background: #0d1117;
                border: 1px solid #30363d;
                color: white;
                border-radius: 4px;
                width: 100%;
            }
            button { background: #238636; cursor: pointer; font-weight: bold; }
            button:hover { background: #2ea043; }
            label { display: block; margin-top: 10px; color: #8b949e; }
            #pred-result { font-size: 24px; color: #58a6ff; margin-top: 15px; text-align: center; }

            .loader { color: #58a6ff; animation: blink 1s infinite; }
            @keyframes blink { 50% { opacity: 0; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>>> TELEGRAM VIEW PREDICTOR SYSTEM</h1>

            <div id="console">
                <div class="log-line INFO">System initialized. Waiting for connection...</div>
            </div>

            <div id="control-panel">
                <h2 style="margin-top:0">Manual Prediction Test</h2>
                <div style="display: flex; gap: 10px;">
                    <div style="flex: 1;">
                        <label>CPM (Cost)</label>
                        <input type="number" id="cpm" value="10.5" step="0.1">
                    </div>
                    <div style="flex: 1;">
                        <label>Date (YYYY-MM-DD)</label>
                        <input type="text" id="date" value="2025-10-05">
                    </div>
                </div>
                <label>Channel Name</label>
                <input type="text" id="channel" value="crypto_news_daily">

                <button onclick="makePrediction()">PREDICT VIEWS</button>
                <div id="pred-result"></div>
            </div>
        </div>

        <script>
            let isTraining = false;
            const consoleDiv = document.getElementById('console');
            const controlPanel = document.getElementById('control-panel');

            // Функция опроса логов
            async function fetchLogs() {
                try {
                    const response = await fetch('/get_logs');
                    const logs = await response.json();

                    consoleDiv.innerHTML = '';
                    logs.forEach(log => {
                        const line = document.createElement('div');
                        line.className = `log-line ${log.level}`;
                        line.innerText = log.msg;
                        consoleDiv.appendChild(line);
                    });
                    consoleDiv.scrollTop = consoleDiv.scrollHeight;

                    // Если увидели успех обучения, показываем панель
                    if (logs.some(l => l.msg.includes("Модель готова к работе"))) {
                        controlPanel.style.display = 'block';
                    }
                } catch (e) { console.error(e); }
            }

            // Запуск обучения при старте
            async function startAutoTrain() {
                setInterval(fetchLogs, 1000); // Опрос логов каждую секунду

                try {
                    await fetch('/train', { method: 'POST' });
                } catch (e) {
                    // Ошибки поймает логгер
                }
            }

            // Функция предикта с кнопки
            async function makePrediction() {
                const btn = document.querySelector('button');
                btn.innerText = "CALCULATING...";

                const data = {
                    cpm: parseFloat(document.getElementById('cpm').value),
                    channel: document.getElementById('channel').value,
                    date: document.getElementById('date').value
                };

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    const json = await res.json();
                    if (json.predicted_views !== undefined) {
                        document.getElementById('pred-result').innerText = `Result: ${json.predicted_views} views`;
                    } else {
                        document.getElementById('pred-result').innerText = `Error: ${json.detail}`;
                    }
                } catch (e) {
                    alert(e);
                }
                btn.innerText = "PREDICT VIEWS";
            }

            // Start
            window.onload = startAutoTrain;
        </script>
    </body>
    </html>
    """


# --- API ENDPOINTS ---

@app.get("/get_logs")
def get_logs():
    """Возвращает текущий буфер логов для JS"""
    return LOG_BUFFER


@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    """Эндпоинт, который запускает обучение."""

    # Очищаем логи перед новым запуском
    LOG_BUFFER.clear()

    # Запускаем логику синхронно (чтобы пользователь видел процесс step-by-step),
    # или можно в background, но тогда сложнее ловить ошибки сразу.
    # Для демо оставим синхронно, но с перехватом всех ошибок.

    try:
        backend_log("=== STARTING TRAINING PIPELINE ===", "INFO")

        # 1. Check files
        if not os.path.exists('train_hack.csv'):
            backend_log("CRITICAL: train_hack.csv не найден!", "ERROR")
            raise HTTPException(status_code=400, detail="No train file")

        backend_log("Чтение CSV файлов...", "INFO")
        train_df = pd.read_csv('train_hack.csv', skipinitialspace=True)
        test_df = pd.read_csv('test_hack.csv', skipinitialspace=True)

        extra_df = pd.DataFrame(columns=train_df.columns)
        if os.path.exists('doc-1769013311.csv'):
            backend_log("Найден дополнительный датасет doc-1769013311.csv. Используем.", "INFO")
            extra_df = pd.read_csv('doc-1769013311.csv', skipinitialspace=True)
        else:
            backend_log("Доп. датасет не найден, пропускаем.", "WARNING")

        # Cleanup
        for df in [train_df, test_df, extra_df]:
            df.columns = df.columns.str.strip()

        # Preprocess
        backend_log("Предобработка дат и чисел...", "INFO")
        train_df = preprocess(train_df)
        test_df = preprocess(test_df)
        extra_df = preprocess(extra_df)
        y = np.log1p(train_df['VIEWS'])

        # NLP
        backend_log("NLP: Создание словаря TF-IDF (может занять время)...", "INFO")
        all_texts = pd.concat([train_df['text_channel'], test_df['text_channel'], extra_df['text_channel']])

        tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=10000, min_df=2)
        tfidf.fit(all_texts)

        backend_log("NLP: Сжатие векторов (SVD)...", "INFO")
        svd = TruncatedSVD(n_components=100, random_state=42)
        svd.fit(tfidf.transform(all_texts))

        def get_vectors(text_series):
            return svd.transform(tfidf.transform(text_series))

        backend_log("Генерация признаков для обучения...", "INFO")
        X_text_train = get_vectors(train_df['text_channel'])
        svd_cols = [f'svd_{i}' for i in range(100)]
        X_train_svd = pd.DataFrame(X_text_train, columns=svd_cols, index=train_df.index)

        num_cols = ['CPM', 'log_cpm', 'month', 'day_of_week', 'day', 'is_weekend']
        X_train = pd.concat([train_df[num_cols], X_train_svd], axis=1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Training Models
        backend_log("Обучение модели KNN (поиск соседей)...", "INFO")
        knn = KNeighborsRegressor(n_neighbors=10, weights='distance', metric='cosine')
        knn.fit(X_train_scaled, y)

        backend_log("Обучение модели Ridge (линейная регрессия)...", "INFO")
        ridge = Ridge(alpha=0.5)
        ridge.fit(X_train_scaled, y)

        backend_log("Обучение модели CatBoost (это самый долгий этап)...", "INFO")
        cb = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=7, loss_function='RMSE', verbose=0,
                               allow_writing_files=False, random_seed=42)
        cb.fit(X_train, y)

        # Save
        backend_log("Сохранение артефактов в models_pro.pkl...", "INFO")
        new_artifacts = {
            "catboost": cb, "ridge": ridge, "knn": knn,
            "scaler": scaler, "tfidf": tfidf, "svd": svd
        }

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(new_artifacts, f)

        load_artifacts()  # Reload in memory

        backend_log("=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===", "SUCCESS")
        backend_log("Модель готова к работе. Используйте форму ниже.", "SUCCESS")

        return {"status": "success"}

    except Exception as e:
        err_msg = traceback.format_exc()
        backend_log(f"КРИТИЧЕСКАЯ ОШИБКА:\n{err_msg}", "ERROR")
        raise HTTPException(status_code=500, detail=str(e))


class AdRequest(BaseModel):
    cpm: float
    channel: str
    date: str


class AdResponse(BaseModel):
    predicted_views: int


@app.post("/predict", response_model=AdResponse)
def predict(req: AdRequest):
    global ARTIFACTS

    if ARTIFACTS is None:
        if not load_artifacts():
            raise HTTPException(status_code=503, detail="Model not loaded. Check console logs.")

    try:
        cb = ARTIFACTS["catboost"]
        ridge = ARTIFACTS["ridge"]
        knn = ARTIFACTS["knn"]
        scaler = ARTIFACTS["scaler"]
        tfidf = ARTIFACTS["tfidf"]
        svd = ARTIFACTS["svd"]

        dt = pd.to_datetime(req.date)

        # Feature Gen
        text = [str(req.channel).lower()]
        vec = svd.transform(tfidf.transform(text))[0]

        feats = [
            req.cpm, np.log1p(req.cpm),
            dt.month, dt.dayofweek, dt.day,
            1 if dt.dayofweek >= 5 else 0
        ]
        feats.extend(vec)

        X_arr = np.array(feats).reshape(1, -1)
        X_scaled = scaler.transform(X_arr)

        p1 = cb.predict(X_arr)[0]
        p2 = ridge.predict(X_scaled)[0]
        p3 = knn.predict(X_scaled)[0]

        final_log = 0.5 * p1 + 0.3 * p3 + 0.2 * p2
        views = int(max(0, np.expm1(final_log)))

        return {"predicted_views": views}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Запускаем сервер
    # reload=False важно, чтобы не сбрасывать глобальные переменные при изменении файлов
    uvicorn.run(app, host="0.0.0.0", port=8000)