import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MultiLingualTextEmbedder:
    """Класс для получения эмбеддингов текста через SentenceTransformer"""
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2', batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def fit(self, X=None):
        if self.model is None:
            logger.info(f"Загрузка модели-эмбеддера: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if not isinstance(X, (list, pd.Series, np.ndarray)):
            X = list(X)
        X_clean = [str(t).strip() if isinstance(t, str) and len(t.strip()) > 0 else " " for t in X]
        embeddings = self.model.encode(
            X_clean,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    def get_dimension(self):
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала необходимо вызвать .fit()")
        return self.model.get_sentence_embedding_dimension()


def remove_highly_correlated_features(df, threshold=0.8):
    """Удаляет признаки с коэффициентом корреляции Пирсона выше порога"""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)
    if to_drop:
        logger.info(f"Удалено {len(to_drop)} сильно коррелирующих признаков (corr>{threshold}): {to_drop}")
    else:
        logger.info("Сильно коррелирующих признаков не обнаружено.")
    return df_reduced, to_drop


def add_track_features(df, track_name_col='track_name'):
    """
    Добавляет признаки на основе названия трека:
    - is_remix: содержит 'remix'
    - is_version: содержит 'version' или 'ver.'
    - is_live: содержит 'live'
    """
    if track_name_col not in df.columns:
        logger.warning(f"Столбец '{track_name_col}' не найден")
        return df

    df['is_remix'] = df[track_name_col].str.lower().str.contains(r'remix', regex=False).astype(int)
    df['is_version'] = df[track_name_col].str.lower().str.contains(r'version|ver\.', regex=True).astype(int)
    df['is_live'] = df[track_name_col].str.lower().str.contains(r'live', regex=False).astype(int)

    logger.info(f"Добавлены признаки: is_remix, is_version, is_live")
    return df


def add_lyrics_features(df, text_col='lyrics'):
    """
    Добавляет признаки по тексту песни:
    - lyrics_word_count: количество слов (разбиение по пробелам)
    - lyrics_line_count: количество строк (по переносам)
    """
    if text_col not in df.columns:
        logger.warning(f"Столбец '{text_col}' не найден, пропускаем создание признаков текста.")
        return df

    # Заполняем NaN пустой строкой для безопасного подсчета
    lyrics_series = df[text_col].fillna('')
    df['lyrics_word_count'] = lyrics_series.str.split().str.len()
    df['lyrics_line_count'] = lyrics_series.str.count('\n') + 1

    logger.info("Добавлены признаки текста: lyrics_word_count, lyrics_line_count")
    return df


def preprocess_data(df, # данные
                    target_col='track_popularity', # целевая переменная
                    text_col='lyrics', # столбец с текстами песен
                    track_name_col='track_name', # столбец с названиями песен
                    id_cols=None, # столбцы идентификаторы для удаления
                    test_size=0.2, # размер тестовой выборки
                    random_state=42, # фиксируем random_state
                    use_text=False, # флажок обработки текстов песен
                    embedder_model='paraphrase-multilingual-mpnet-base-v2', # модель для эмбеддингов (sentence-transformers)
                    save_preprocessor=True, # сохранение scaler и embedder
                    preprocessor_dir='.'): # папка для сохранения
    """
    Функция для предобработки данных:
    1. Удаление дубликатов по id
    2. Удаление созависимых с целевой переменной признаков
    3. Добавление признаков по названию и тексту песен
    4. Обработка пропусков в числовых признаках
    5.
    """
    logger.info("Предобработка данных...")

    data = df.copy()

    # Удаление дубликатов
    if id_cols is None:
        id_cols = []
    if 'track_id' in data.columns and 'track_id' not in id_cols:
        id_cols.append('track_id')
    if id_cols:
        # Удаляем дубликаты по всем идентификаторам, если они есть
        dup_cols = [col for col in id_cols if col in data.columns]
        if dup_cols:
            before = len(data)
            data.drop_duplicates(subset=dup_cols, inplace=True)
            logger.info(f"Удалены дубликаты по {dup_cols}, осталось {len(data)} строк (было {before})")

    # Удаление зависимых признаков (popularity)
    popularity_cols = [col for col in data.columns if 'popularity' in col.lower() and col != target_col]
    additional_drop = ['artist_followers']
    to_drop = list(set(popularity_cols + additional_drop))
    to_drop = [col for col in to_drop if col in data.columns]
    if to_drop:
        data.drop(columns=to_drop, inplace=True)
        logger.info(f"Удалены зависимые признаки: {to_drop}")

    # Feature engineering по названию трека
    data = add_track_features(data, track_name_col=track_name_col)

    # Feature engineering по тексту
    data = add_lyrics_features(data, text_col=text_col)

    # Целевая переменная
    if target_col not in data.columns:
        raise ValueError(f"Целевая колонка '{target_col}' не найдена.")
    y = data[target_col]
    X = data.drop(columns=[target_col])

    # Определяем строковые колонки (object)
    string_cols = data.select_dtypes(include=['object']).columns.tolist()
    # Оставляем текстовую колонку, если будем использовать эмбеддинги
    if use_text and text_col in string_cols:
        string_cols.remove(text_col)
    if string_cols:
        data.drop(columns=string_cols, inplace=True)
        logger.info(f"Удалены строковые колонки: {string_cols}")
    # Если use_text=False, удаляем и текстовую колонку, если она есть
    if not use_text and text_col in data.columns:
        data.drop(columns=[text_col], inplace=True)
        logger.info(f"Удалена текстовая колонка '{text_col}' (use_text=False)")

    # Обработка пропусков в числовых признаках
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    # Также удаляем идентификаторы (если они остались)
    for col in id_cols:
        if col in numeric_cols:
            numeric_cols.remove(col)

    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = pd.DataFrame(
            imputer.fit_transform(X[numeric_cols]),
            index=X.index,
            columns=numeric_cols
        )
        logger.info(f"Пропуски в числовых столбцах заполнены медианой (столбцов: {len(numeric_cols)})")

    # Удаление сильно коррелированных числовых признаков
    if len(numeric_cols) > 1:
        X, _ = remove_highly_correlated_features(X[numeric_cols], threshold=0.9)
        # Обновляем список числовых колонок после удаления
        numeric_cols = X.columns.tolist()
    else:
        numeric_cols = numeric_cols

    # Масштабирование числовых признаков
    scaler = StandardScaler()
    X_scaled = X.copy()
    if len(numeric_cols) > 0:
        X_scaled[numeric_cols] = pd.DataFrame(
            scaler.fit_transform(X_scaled[numeric_cols]),
            index=X_scaled.index,
            columns=numeric_cols
        )
        logger.info(f"Числовые признаки масштабированы (столбцов: {len(numeric_cols)})")
        if save_preprocessor:
            joblib.dump(scaler, os.path.join(preprocessor_dir, 'scaler.pkl'))
    else:
        logger.warning("Нет числовых признаков для масштабирования.")

    # Создание эмбеддингов
    embedder = None
    if use_text and text_col in data.columns:
        logger.info("Создание эмбеддингов...")
        embedder = MultiLingualTextEmbedder(model_name=embedder_model)
        embedder.fit()
        text_embeddings = embedder.transform(data[text_col].fillna(''))
        emb_df = pd.DataFrame(
            text_embeddings,
            index=data.index,
            columns=[f'embed_{i}' for i in range(text_embeddings.shape[1])]
        )
        X_final = pd.concat([X_scaled, emb_df.loc[X_scaled.index]], axis=1)
        logger.info(f"Добавлены эмбеддинги размерности {text_embeddings.shape[1]}")
        if save_preprocessor:
            joblib.dump(embedder, os.path.join(preprocessor_dir, 'embedder.pkl'))
    else:
        X_final = X_scaled
        if use_text:
            logger.warning(f"Столбец '{text_col}' не найден, пропускаем векторизацию")

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Разделение на обучающую и тестовую выборки: train {X_train.shape[0]}, test {X_test.shape[0]}")

    # Сохраняем названия признаков для интерпретации
    feature_names = X_final.columns.tolist()
    if save_preprocessor:
        joblib.dump(feature_names, os.path.join(preprocessor_dir, 'feature_names.pkl'))

    logger.info("Предобработка завершена")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'embedder': embedder,
        'feature_names': feature_names
    }


# Пример использования
if __name__ == '__main__':

    example_df = pd.read_csv('checkpoint_9000.csv')

    result = preprocess_data(
        example_df,
        target_col='track_popularity',
        text_col='lyrics',
        track_name_col='track_name',
        id_cols=['track_id'],
        use_text=True,
        embedder_model='paraphrase-multilingual-mpnet-base-v2',
        save_preprocessor=True,
        preprocessor_dir='.'
    )

    print(f"Получены обучающие данные:\nX_train: {result['X_train'].shape}, X_test: {result['X_test'].shape}")
    print(f"Добавлены признаки: {[col for col in result['X_train'].columns if 'is_' in col or 'lyrics_' in col]}")