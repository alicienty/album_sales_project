# Прогнозирование популярности K-pop треков (Spotify)

## Описание

Проект направлен на **предсказание популярности треков жанра K-pop** (`track_popularity`) на основе аудио-характеристик, текстов песен и других метаданных. Цель проекта — выявить ключевые факторы, влияющие на успех трека, и построить модель, способную прогнозировать популярность новых релизов.

---

## Задачи

- Сбор и предобработка данных о треках Spotify
- Добавление текстов песен (парсинг) и создание мультиязычных эмбеддингов (`sentence-transformers`)
- Feature engeneering (создание признаков по жанрам, названию и тексту песен)
- Обучение и сравнение 5 моделей машинного обучения
- Интерпретация моделей с помощью SHAP и LIME

---

## Обучаемые модели

- **Linear Regression** (базовая модель)
- **Random Forest**
- **SVM** (Support Vector Machine)
- **XGBoost**
- **CatBoost**

---

## Используемые признаки

### 1. Аудио-характеристики
- `tempo` — темп (BPM)
- `key` — тональность
- `mode` — мажор/минор
- `danceability` — танцевальность
- `energy` — энергичность
- `loudness` — громкость
- `speechiness` — речевая составляющая
- `acousticness` — акустичность
- `instrumentalness` — инструментальность
- `liveness` — живость
- `valence` — позитивность
- `energy_danceability_score` — комбинированный показатель

### 2. Инженерные признаки
- `gender_type` — пол исполнителя/участников группы
- `is_group` — группа/сольный артист
- `is_remix` — содержит ли название трека слово "remix"
- `is_version` — содержит ли название "version" или "ver."
- `is_live` — содержит ли название "live"
- `lyrics_word_count` — количество слов в тексте
- `lyrics_line_count` — количество строк в тексте

### 3. Текстовые эмбеддинги
- Используется модель `paraphrase-multilingual-mpnet-base-v2` (размерность 768)
- Альтернатива: `all-MiniLM-L6-v2` (размерность 384, быстрее)

---

## Структура проекта

```
spotify-track-popularity-prediction/
│
├── README.md                          # Описание проекта
├── requirements.txt                   # Зависимости
├── .gitignore                         # Игнорируемые файлы
│
├── preprocess.py                      # Предобработка данных + feature engineering
├── train.py                           # Обучение моделей
├── interpret.py                       # Интерпретация (SHAP + LIME)
│
├── checkpoints/                       # Сохраненные модели и препроцессоры
│   ├── best_model_numeric.pkl         # Лучшая модель (числовые признаки)
│   ├── best_model_text.pkl            # Лучшая модель (числовые + текст)
│   ├── scaler.pkl                     # Масштабатор числовых признаков
│   ├── embedder.pkl                   # Модель эмбеддингов
│   └── feature_names.pkl              # Список признаков
│
├── reports/                           # Отчеты и графики
│   ├── figures/                       # Графики (SHAP, LIME)
│   │   ├── shap_summary_*.png
│   │   ├── shap_waterfall_*.png
│   │   ├── lime_explanation_*.png
│   │   └── interpretation_summary.json
│   ├── results_numeric_only.csv       # Результаты (числовые признаки)
│   └── results_with_text.csv          # Результаты (числовые + текст)
│
└── data/                              # Данные (не коммитятся)
    └── lyrics_11k.csv                   # Данные с текстами песен
```

---

## Установка и настройка

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/spotify-track-popularity-prediction.git
cd spotify-track-popularity-prediction
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# или
venv\Scripts\activate             # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка данных

Поместите файл с данными (например, `df_meta_ml.csv` или `lyrics_10k.csv`) в корневую папку проекта.

---

## Использование

### 1. Предобработка данных

```python
from preprocess import preprocess_data

result = preprocess_data(
    df=df,
    target_col='track_popularity',
    text_col='lyrics',
    track_name_col='track_name',
    id_cols=['track_id'],
    use_text=True,                          # Включить текстовые эмбеддинги
    embedder_model='paraphrase-multilingual-mpnet-base-v2',
    save_preprocessor=True,
    preprocessor_dir='.'
)

X_train, X_test, y_train, y_test = result['X_train'], result['X_test'], result['y_train'], result['y_test']
```

### 2. Обучение моделей

```bash
python train.py
```

Скрипт выполнит два эксперимента:
- **Только числовые признаки** (эксперимент 1)
- **Числовые + текстовые эмбеддинги** (эксперимент 2)

Результаты сохраняются в папку `reports/`:
- `results_numeric_only.csv` — метрики для числовых признаков
- `results_with_text.csv` — метрики для числовых + текст
- `best_model_numeric.pkl` / `best_model_text.pkl` — лучшие модели

### 3. Интерпретация моделей (SHAP + LIME)

```bash
python interpret.py df_meta_ml.csv best_model.pkl false reports/figures 2000 20
```

**Параметры:**
1. `data_path` — путь к данным (CSV)
2. `model_path` — путь к сохраненной модели (.pkl)
3. `use_text` — `true` или `false` (использовались ли эмбеддинги)
4. `output_dir` — папка для сохранения графиков
5. `sample_size` — максимальное число строк для ускорения (по умолчанию 1000)
6. `top_k` — число признаков для отображения (по умолчанию 20)

В папке `reports/figures/` появятся:
- `shap_summary_*.png` — глобальная важность признаков
- `shap_waterfall_*.png` — объяснение отдельных предсказаний
- `lime_explanation_*.png` — LIME-объяснения
- `interpretation_summary.json` — сводка результатов

---

## Метрики оценки моделей

Для оценки качества моделей используются следующие метрики:

| Метрика | Что измеряет | Интерпретация |
| :--- | :--- | :--- |
| **MSE** (Mean Squared Error) | Средний квадрат ошибки | Чем меньше — тем лучше. Чувствительна к большим ошибкам. |
| **RMSE** (Root Mean Squared Error) | Корень из MSE | Ошибка в единицах целевой переменной (`track_popularity`). |
| **MAE** (Mean Absolute Error) | Средняя абсолютная ошибка | Среднее отклонение предсказания от реального значения. |
| **R²** (Коэффициент детерминации) | Доля объясненной дисперсии | От 0 до 1. Чем ближе к 1 — тем лучше. |

---

## Результаты экспериментов

### Эксперимент 1: Только числовые признаки

| Модель | MSE | RMSE | MAE | R² |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression | 237.70 | 15.42 | 12.33 | 0.0714 |
| Random Forest | 216.55 | 14.72 | 11.60 | 0.1540 |
| SVM | 234.22 | 15.30 | 11.75 | 0.0850 |
| **XGBoost** | 211.89 | 14.56 | 11.45 | 0.1722 |
| **CatBoost** | 209.06 | 14.46 | 11.40 | 0.1833 |

### Эксперимент 2: Числовые + текстовые эмбеддинги

| Модель | MSE | RMSE | MAE | R² |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 233.3992 | 15.2774 | 12.0146 | 0.0882 |
| **Random Forest** | 215.5949 | 14.6832 | 11.6904 | 0.1577 |
| **SVM** | 221.2485 | 14.8744 | 11.3364 | 0.1357 |
| **XGBoost** | 213.0030 | 14.5946 | 11.5677 | 0.1679 |
| **CatBoost** | 209.3873 | 14.4702 | 11.4641 | 0.1820 |

---

## Интерпретация результатов с SHAP и LIME

Проект включает детальную интерпретацию моделей:

### SHAP (SHapley Additive exPlanations)
- **Summary Plot** — глобальная важность признаков
- **Waterfall Plot** — объяснение каждого предсказания
- **Dependence Plot** — влияние отдельных признаков на предсказание

### LIME (Local Interpretable Model-agnostic Explanations)
- Объяснение конкретных предсказаний
- Агрегированная важность признаков по нескольким примерам

---

## Использование ИИ в проекте


**DeepSeek** — для:
   - Генерации структуры проекта и архитектуры
   - Написания шаблонов кода для preprocess.py, train.py, interpret.py
   - Отладки ошибок и оптимизации кода
   - Создания документации и README

**Промпты для ИИ:**
```
"Напиши функцию предобработки данных для предсказания track_popularity с обработкой пропусков, удалением коррелированных признаков и масштабированием"
"Сгенерируй значения гиперпараметров моделей для RandomizedSearch"
"Предложи идеи для модуля интерпретации моделей с использованием SHAP и LIME"
"Объясни значения метрик MSE, RMSE, MAE, R² для результатов обучения"
```

---

## Вклад участников

| Участник | Роль | Вклад |
| :--- | :--- | :--- |
| **@polly842** | Team Lead, ML Engineer | Обучение моделей, подбор гиперпараметров, настройка пайплайна, тестирование |
| **@aliences** | Data Engineer, документация/тесты | Сбор данных, предобработка, feature engineering, интерпретация, написание документации |

---

Проект вдохновлен исследованием:

- Kim J., Kim J., Seo M., Song J. *Analysis of K-pop album sales using machine learning models* // Korean J Appl Stat, 2025, 38(2), p. 227-249.  
  DOI: [https://doi.org/10.5351/KJAS.2025.38.2.227](https://doi.org/10.5351/KJAS.2025.38.2.227)
