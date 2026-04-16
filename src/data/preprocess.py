"""
Модуль предобработки данных для проекта прогнозирования продаж K-pop альбомов

Выполняет:
- Обработку пропусков (KNN Imputation)
- Удаление дубликатов
- Кодирование категориальных переменных (One-Hot Encoding)
- Масштабирование числовых признаков (StandardScaler)
- Обработку выбросов
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(file_path='../data/raw/kpop_albums_raw.csv'):
    """
    Загрузка сырых данных
    
    Parameters:
    -----------
    file_path : str
        Путь к файлу с сырыми данными
        
    Returns:
    --------
    pd.DataFrame
        Загруженный датафрейм
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
        return df
    except FileNotFoundError:
        print(f"❌ Файл не найден: {file_path}")


def explore_missing_values(df):
    """
    Анализ пропущенных значений
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный датафрейм
        
    Returns:
    --------
    pd.DataFrame
        DataFrame с информацией о пропусках
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing_count.values,
        'missing_percent': missing_percent.values
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_percent', ascending=False)
    
    print("\n" + "="*50)
    print("АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
    print("="*50)
    
    if len(missing_df) == 0:
        print("✅ Пропущенных значений нет!")
    else:
        print(missing_df.to_string(index=False))
        print(f"\n📊 Всего столбцов с пропусками: {len(missing_df)}")
    
    return missing_df


def handle_missing_values(df, categorical_cols, numerical_cols, strategy='knn'):
    """
    Обработка пропущенных значений
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный датафрейм
    categorical_cols : list
        Список категориальных колонок
    numerical_cols : list
        Список числовых колонок
    strategy : str
        Стратегия заполнения ('knn', 'mean', 'median', 'mode')
        
    Returns:
    --------
    pd.DataFrame
        Датафрейм с заполненными пропусками
    """
    df_clean = df.copy()
    
    print("\n" + "="*50)
    print(f"ОБРАБОТКА ПРОПУСКОВ (стратегия: {strategy})")
    print("="*50)
    
    # Числовые колонки
    num_cols_existing = [col for col in numerical_cols if col in df_clean.columns]
    
    if strategy == 'knn' and len(num_cols_existing) > 0:
        # KNN Imputation (как в оригинальной статье)
        imputer = KNNImputer(n_neighbors=5)
        df_clean[num_cols_existing] = imputer.fit_transform(df_clean[num_cols_existing])
        print(f"✅ KNN Imputation применен к {len(num_cols_existing)} числовым колонкам")
    
    elif strategy == 'mean':
        for col in num_cols_existing:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        print(f"✅ Заполнение средним для {len(num_cols_existing)} числовых колонок")
    
    elif strategy == 'median':
        for col in num_cols_existing:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        print(f"✅ Заполнение медианой для {len(num_cols_existing)} числовых колонок")
    
    # Категориальные колонки
    cat_cols_existing = [col for col in categorical_cols if col in df_clean.columns]
    
    for col in cat_cols_existing:
        mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'unknown'
        df_clean[col].fillna(mode_value, inplace=True)
    
    print(f"✅ Заполнение модой для {len(cat_cols_existing)} категориальных колонок")
    
    # Проверяем остались ли пропуски
    remaining_missing = df_clean.isnull().sum().sum()
    if remaining_missing == 0:
        print("✅ Все пропуски успешно обработаны!")
    else:
        print(f"⚠️ Осталось пропусков: {remaining_missing}")
    
    return df_clean


def remove_duplicates_and_invalid(df, id_col='album_id'):
    """
    Удаление дубликатов и некорректных записей
    
    Parameters:
    -----------
    df : pd.DataFrame
        Датафрейм для очистки
    id_col : str
        Название колонки с уникальным идентификатором
        
    Returns:
    --------
    pd.DataFrame
        Очищенный датафрейм
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    print("\n" + "="*50)
    print("УДАЛЕНИЕ ДУБЛИКАТОВ И НЕКОРРЕКТНЫХ ЗАПИСЕЙ")
    print("="*50)
    
    # Удаление дубликатов по ID
    if id_col in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=[id_col])
        print(f"✅ Удалено дубликатов по {id_col}: {initial_rows - len(df_clean)}")
    
    # Удаление строк с отрицательными значениями в числовых колонках
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != id_col:
            df_clean = df_clean[df_clean[col] >= 0]
    
    after_negatives = len(df_clean)
    print(f"✅ Удалено строк с отрицательными значениями: {initial_rows - after_negatives}")
    
    # Удаление выбросов по целевой переменной (опционально)
    if 'first_week_sales' in df_clean.columns:
        # Оставляем только реалистичные значения
        q99 = df_clean['first_week_sales'].quantile(0.99)
        df_clean = df_clean[df_clean['first_week_sales'] <= q99 * 1.5]
        print(f"✅ Удалено выбросов по целевой переменной: {after_negatives - len(df_clean)}")
    
    print(f"\n📊 Итоговое количество строк: {len(df_clean)} (было: {initial_rows})")
    
    return df_clean


def encode_categorical_variables(df, categorical_cols, handle_unknown='ignore'):
    """
    One-Hot Encoding для категориальных переменных
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный датафрейм
    categorical_cols : list
        Список категориальных колонок для кодирования
    handle_unknown : str
        Стратегия обработки неизвестных категорий
        
    Returns:
    --------
    tuple (pd.DataFrame, OneHotEncoder)
        Закодированный датафрейм и обученный encoder
    """
    df_encoded = df.copy()
    
    # Фильтруем только существующие колонки
    cat_cols_existing = [col for col in categorical_cols if col in df_encoded.columns]
    
    if len(cat_cols_existing) == 0:
        print("⚠️ Нет категориальных колонок для кодирования")
        return df_encoded, None
    
    print("\n" + "="*50)
    print("ONE-HOT ENCODING")
    print("="*50)
    print(f"Кодируемые колонки: {cat_cols_existing}")
    
    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
    encoded_array = encoder.fit_transform(df_encoded[cat_cols_existing])
    
    # Создаем DataFrame с закодированными признаками
    encoded_cols = []
    for i, col in enumerate(cat_cols_existing):
        for category in encoder.categories_[i]:
            encoded_cols.append(f"{col}_{category}")
    
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_cols,
        index=df_encoded.index
    )
    
    # Удаляем исходные категориальные колонки и добавляем закодированные
    df_encoded = df_encoded.drop(columns=cat_cols_existing)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    print(f"✅ Закодировано {len(cat_cols_existing)} колонок -> {len(encoded_cols)} новых признаков")
    
    return df_encoded, encoder


def scale_numerical_features(df, numerical_cols, scaler_type='standard'):
    """
    Масштабирование числовых признаков
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный датафрейм
    numerical_cols : list
        Список числовых колонок для масштабирования
    scaler_type : str
        Тип скейлера ('standard', 'robust')
        
    Returns:
    --------
    tuple (pd.DataFrame, StandardScaler/RobustScaler)
        Масштабированный датафрейм и обученный scaler
    """
    df_scaled = df.copy()
    
    # Фильтруем только существующие числовые колонки
    num_cols_existing = [col for col in numerical_cols if col in df_scaled.columns]
    
    if len(num_cols_existing) == 0:
        print("⚠️ Нет числовых колонок для масштабирования")
        return df_scaled, None
    
    print("\n" + "="*50)
    print(f"МАСШТАБИРОВАНИЕ ПРИЗНАКОВ (тип: {scaler_type})")
    print("="*50)
    print(f"Масштабируемые колонки: {num_cols_existing}")
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()  # Устойчив к выбросам
    else:
        raise ValueError(f"Неизвестный тип скейлера: {scaler_type}")
    
    df_scaled[num_cols_existing] = scaler.fit_transform(df_scaled[num_cols_existing])
    
    print(f"✅ Масштабировано {len(num_cols_existing)} признаков")
    
    return df_scaled, scaler


def handle_outliers(df, numerical_cols, method='iqr', threshold=3):
    """
    Обработка выбросов
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный датафрейм
    numerical_cols : list
        Список числовых колонок
    method : str
        Метод обработки ('clip', 'remove', 'cap')
    threshold : float
        Порог для определения выбросов
        
    Returns:
    --------
    pd.DataFrame
        Датафрейм с обработанными выбросами
    """
    df_clean = df.copy()
    
    print("\n" + "="*50)
    print(f"ОБРАБОТКА ВЫБРОСОВ (метод: {method})")
    print("="*50)
    
    num_cols_existing = [col for col in numerical_cols if col in df_clean.columns]
    
    outliers_count = 0
    
    for col in num_cols_existing:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            outliers_count += outliers
            
            if method == 'clip':
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            elif method == 'cap':
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        
        elif method == 'zscore':
            zscore = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
            outliers = (zscore > threshold).sum()
            outliers_count += outliers
            
            if method == 'clip':
                df_clean.loc[zscore > threshold, col] = df_clean[col].median()
    
    if method == 'remove':
        # Удаляем строки с выбросами
        for col in num_cols_existing:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"✅ Удалено строк с выбросами: {len(df) - len(df_clean)}")
    else:
        print(f"✅ Обработано выбросов: {outliers_count}")
    
    return df_clean


def preprocess_data(
    df=None,
    target_col='first_week_sales',
    id_col='album_id',
    categorical_cols=None,
    numerical_cols=None,
    missing_strategy='knn',
    scaling_type='standard',
    encode_categorical=True,
    handle_outliers_method=None,
    return_scalers=False
):
    """
    ГЛАВНАЯ ФУНКЦИЯ ПРЕДОБРАБОТКИ ДАННЫХ
    
    Полный пайплайн обработки данных для проекта
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        Исходный датафрейм. Если None, загружает из файла
    target_col : str
        Название целевой переменной
    id_col : str
        Название колонки с ID
    categorical_cols : list, optional
        Список категориальных колонок (если None - определяет автоматически)
    numerical_cols : list, optional
        Список числовых колонок (если None - определяет автоматически)
    missing_strategy : str
        Стратегия заполнения пропусков ('knn', 'mean', 'median')
    scaling_type : str
        Тип масштабирования ('standard', 'robust')
    encode_categorical : bool
        Применять ли One-Hot Encoding
    handle_outliers_method : str, optional
        Метод обработки выбросов ('clip', 'remove', 'cap') или None
    return_scalers : bool
        Возвращать ли обученные scaler и encoder
        
    Returns:
    --------
    dict
        Словарь с обработанными данными и метаинформацией
    """
    
    print("\n" + "="*70)
    print(" " * 20 + "ПРЕДОБРАБОТКА ДАННЫХ")
    print("="*70)
    
    # 1. Загрузка данных
    if df is None:
        df = load_raw_data()
    
    # 2. Определение типов колонок
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Убираем ID и целевую переменную
        categorical_cols = [col for col in categorical_cols if col not in [id_col, target_col]]
    
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Убираем ID и целевую переменную
        numerical_cols = [col for col in numerical_cols if col not in [id_col, target_col]]
    
    print(f"\n📋 Определено категориальных колонок: {len(categorical_cols)}")
    print(f"📋 Определено числовых колонок: {len(numerical_cols)}")
    
    # 3. Анализ пропусков
    explore_missing_values(df)
    
    # 4. Обработка пропусков
    df = handle_missing_values(df, categorical_cols, numerical_cols, strategy=missing_strategy)
    
    # 5. Удаление дубликатов и некорректных записей
    df = remove_duplicates_and_invalid(df, id_col=id_col)
    
    # 6. Обработка выбросов (опционально)
    if handle_outliers_method:
        df = handle_outliers(df, numerical_cols, method=handle_outliers_method)
    
    # 7. One-Hot Encoding
    encoder = None
    if encode_categorical:
        df, encoder = encode_categorical_variables(df, categorical_cols)
    
    # 8. Масштабирование
    scaler = None
    df, scaler = scale_numerical_features(df, numerical_cols, scaler_type=scaling_type)
    
    # 9. Разделение на признаки и целевую переменную
    if target_col in df.columns:
        X = df.drop(columns=[target_col, id_col] if id_col in df.columns else [target_col])
        y = df[target_col]
        
        # Логарифмическое преобразование целевой переменной (как в статье)
        y_log = np.log1p(y)  # log(1 + x) для устойчивости
        
        print(f"\n📊 Итоговая размерность X: {X.shape}")
        print(f"📊 Итоговая размерность y: {y.shape}")
    else:
        X = df.drop(columns=[id_col] if id_col in df.columns else [])
        y = None
        y_log = None
        print("\n⚠️ Целевая переменная не найдена")
    
    # 10. Результаты
    results = {
        'X': X,
        'y': y,
        'y_log': y_log,
        'feature_names': X.columns.tolist(),
        'original_df_shape': df.shape,
        'categorical_cols_original': categorical_cols,
        'numerical_cols_original': numerical_cols,
        'target_col': target_col,
        'id_col': id_col
    }
    
    if return_scalers:
        results['encoder'] = encoder
        results['scaler'] = scaler
    
    print("\n" + "="*70)
    print(" " * 25 + "ПРЕДОБРАБОТКА ЗАВЕРШЕНА")
    print("="*70)
    
    return results


def create_preprocessing_pipeline(categorical_cols, numerical_cols, scaling_type='standard'):
    """
    Создание sklearn Pipeline для предобработки
    
    Parameters:
    -----------
    categorical_cols : list
        Список категориальных колонок
    numerical_cols : list
        Список числовых колонок
    scaling_type : str
        Тип масштабирования
        
    Returns:
    --------
    ColumnTransformer
        Пайплайн для предобработки
    """
    
    # Preprocessing for numerical columns
    if scaling_type == 'standard':
        num_transformer = StandardScaler()
    else:
        num_transformer = RobustScaler()
    
    # Preprocessing for categorical columns
    cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ]
    )
    
    return preprocessor


# Пример использования
if __name__ == "__main__":
    # Демонстрация работы функции
    
    # 1. Создаем пример данных
    df = create_sample_data()
    
    # 2. Запускаем предобработку
    results = preprocess_data(
        df=df,
        target_col='first_week_sales',
        id_col='album_id',
        missing_strategy='knn',
        scaling_type='standard',
        encode_categorical=True,
        handle_outliers_method='clip',
        return_scalers=True
    )
    
    # 3. Проверяем результаты
    print("\n" + "="*50)
    print("ПРОВЕРКА РЕЗУЛЬТАТОВ")
    print("="*50)
    print(f"X shape: {results['X'].shape}")
    print(f"y shape: {results['y'].shape}")
    print(f"y_log shape: {results['y_log'].shape}")
    print(f"Feature names ({len(results['feature_names'])}): {results['feature_names'][:5]}...")
    
    # 4. Сохраняем обработанные данные
    import os
    os.makedirs('../data/processed', exist_ok=True)
    
    results['X'].to_csv('../data/processed/X_data.csv', index=False)
    pd.DataFrame({target_col: results['y'], f'{target_col}_log': results['y_log']}).to_csv(
        '../data/processed/y_data.csv', index=False
    )
    print("\n✅ Обработанные данные сохранены в 'data/processed/'")
