import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

# 1. ЗАГРУЗКА ВСЕХ ДАННЫХ
print(">>> Загрузка данных...")
train_df = pd.read_csv('train_hack.csv', skipinitialspace=True)
test_df = pd.read_csv('test_hack.csv', skipinitialspace=True)
extra_df = pd.read_csv('doc-1769013311.csv', skipinitialspace=True) # Твой новый файл

# Чистим пробелы
for df in [train_df, test_df, extra_df]:
    df.columns = df.columns.str.strip()

# 2. ПРЕДОБРАБОТКА
def preprocess(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['month'] = df['DATE'].dt.month
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['day'] = df['DATE'].dt.day
    df['is_weekend'] = (df['DATE'].dt.dayofweek >= 5).astype(int)
    df['log_cpm'] = np.log1p(df['CPM'])
    # Текст: все в нижний, замена nan
    df['text_channel'] = df['CHANNEL_NAME'].astype(str).fillna("unknown").str.lower()
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)
extra_df = preprocess(extra_df)

# Таргет (log)
y = np.log1p(train_df['VIEWS'])

# 3. МОЩНЫЙ NLP (TF-IDF + SVD)
print(">>> Обучение NLP на полном словаре (400k+ каналов)...")
# Объединяем тексты из ВСЕХ источников, чтобы знать все слова
all_texts = pd.concat([
    train_df['text_channel'],
    test_df['text_channel'],
    extra_df['text_channel']
])

# char_wb + ngrams=(2,5) идеально ловит похожесть названий
tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 5),
    max_features=10000, # Увеличили словарь
    min_df=2
)
tfidf.fit(all_texts)

# SVD - сжимаем до 100 компонент для точности
svd = TruncatedSVD(n_components=100, random_state=42)
svd.fit(tfidf.transform(all_texts))

print(">>> Трансформация текстов...")
# Функция получения векторов
def get_vectors(text_series):
    return svd.transform(tfidf.transform(text_series))

X_text_train = get_vectors(train_df['text_channel'])
X_text_test = get_vectors(test_df['text_channel'])

# Создаем DF из векторов
svd_cols = [f'svd_{i}' for i in range(100)]
X_train_svd = pd.DataFrame(X_text_train, columns=svd_cols, index=train_df.index)
X_test_svd = pd.DataFrame(X_text_test, columns=svd_cols, index=test_df.index)

# 4. СБОРКА ФИЧЕЙ
num_cols = ['CPM', 'log_cpm', 'month', 'day_of_week', 'day', 'is_weekend']
X_train = pd.concat([train_df[num_cols], X_train_svd], axis=1)
X_test = pd.concat([test_df[num_cols], X_test_svd], axis=1)

# Масштабирование (нужно для KNN и Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. ОБУЧЕНИЕ СТЕКА МОДЕЛЕЙ
print(">>> Обучение ансамбля...")

# Модель 1: KNN (Поиск соседей) - КИЛЛЕР ФИЧА ДЛЯ 0.1
# Ищет 10 каналов с похожими названиями и характеристиками
print("   -> KNN Regressor...")
knn = KNeighborsRegressor(n_neighbors=10, weights='distance', metric='cosine')
knn.fit(X_train_scaled, y)

# Модель 2: Ridge (Линейная база)
print("   -> Ridge...")
ridge = Ridge(alpha=0.5)
ridge.fit(X_train_scaled, y)

# Модель 3: CatBoost (Основная мощь)
print("   -> CatBoost...")
cb = CatBoostRegressor(
    iterations=2000, learning_rate=0.03, depth=7,
    loss_function='RMSE', verbose=0, allow_writing_files=False, random_seed=42
)
cb.fit(X_train, y)

# 6. СОХРАНЕНИЕ
artifacts = {
    "catboost": cb, "ridge": ridge, "knn": knn,
    "scaler": scaler, "tfidf": tfidf, "svd": svd
}
with open("models_pro.pkl", "wb") as f:
    pickle.dump(artifacts, f)
print(">>> Модели сохранены в models_pro.pkl")

# 7. ПРЕДСКАЗАНИЕ И САБМИТ
print(">>> Генерация прогноза...")

p_cb = cb.predict(X_test)
p_ridge = ridge.predict(X_test_scaled)
p_knn = knn.predict(X_test_scaled)

# Взвешенный ансамбль:
# KNN даем большой вес, так как он находит "исторических близнецов"
final_log = 0.5 * p_cb + 0.3 * p_knn + 0.2 * p_ridge

final_pred = np.expm1(final_log)
final_pred = np.maximum(final_pred, 0).astype(int)

submission = test_df.copy()
submission['VIEWS'] = final_pred
submission.to_csv('submission_pro.csv', index=False)
print(">>> Сабмит готов: submission_pro.csv")