import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import joblib

# Импортируем функцию предобработки из модуля
from preprocess import preprocess_data

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_data_and_model(data_path, model_path, use_text=False, sample_size=None):
    """
    Загружает данные, выполняет предобработку и загружает модель.
    Если sample_size задан, берет подвыборку для ускорения интерпретации.

    Возвращает:
        X_train, X_test, y_train, y_test, model, feature_names, scaler, embedder
    """
    # Загрузка данных
    df = pd.read_csv(data_path, index_col=0)
    logger.info(f"Загружено {len(df)} строк, {df.shape[1]} колонок")

    # Предобработка (такая же, как при обучении)
    result = preprocess_data(
        df,
        use_text=use_text,
        test_size=0.2,
        random_state=42,
        save_preprocessor=False
    )
    X_train = result['X_train']
    X_test = result['X_test']
    y_train = result['y_train']
    y_test = result['y_test']
    scaler = result['scaler']
    embedder = result['embedder']
    feature_names = result['feature_names']

    # Если задан sample_size, уменьшаем выборку для ускорения
    if sample_size and len(X_train) > sample_size:
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
        y_train = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        logger.info(f"Обучающая выборка уменьшена до {sample_size} строк")

    if sample_size and len(X_test) > sample_size // 2:
        indices = np.random.choice(len(X_test), sample_size // 2, replace=False)
        X_test = X_test.iloc[indices] if hasattr(X_test, 'iloc') else X_test[indices]
        y_test = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]
        logger.info(f"Тестовая выборка уменьшена до {sample_size // 2} строк")

    # Загрузка модели
    model = joblib.load(model_path)
    logger.info(f"Модель загружена: {type(model).__name__}")

    # Проверка, что X_train – DataFrame, иначе преобразуем
    if not hasattr(X_train, 'columns'):
        # Если X_train – numpy array, создаем имена признаков
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)

    return X_train, X_test, y_train, y_test, model, feature_names, scaler, embedder


def shap_interpretation(model, X_train, X_test, feature_names, model_name, output_dir, top_k=None):
    """
    SHAP: summary, waterfall, dependence plots.
    Если top_k задан, используем только top_k признаков для ускорения.
    Сохраняет графики в output_dir.
    """
    logger.info("** SHAP Analysis **")

    # Определяем explainer в зависимости от типа модели
    model_class = model.__class__.__name__

    try:
        if model_class in ['RandomForestRegressor', 'XGBRegressor', 'CatBoostRegressor']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            logger.info("Используется TreeExplainer")
        elif model_class == 'LinearRegression':
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
            logger.info("Используется LinearExplainer")
        else:
            # Для SVR и других – KernelExplainer с подвыборкой
            sample_size = min(200, len(X_train))
            background = shap.sample(X_train, sample_size)
            explainer = shap.KernelExplainer(model.predict, background)
            test_sample = shap.sample(X_test, min(100, len(X_test)))
            shap_values = explainer.shap_values(test_sample)
            X_test = test_sample
            logger.info("Используется KernelExplainer (с подвыборкой)")
    except Exception as e:
        logger.error(f"Ошибка при создании SHAP explainer: {e}")
        return None

    # Если top_k задан, оставляем только самые важные признаки (по среднему |SHAP|)
    if top_k and top_k < len(feature_names):
        # Вычисляем важность для сортировки
        importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(importance)[-top_k:][::-1]
        shap_values = shap_values[:, top_indices]
        X_test = X_test.iloc[:, top_indices] if hasattr(X_test, 'iloc') else X_test[:, top_indices]
        feature_names = [feature_names[i] for i in top_indices]
        logger.info(f"Оставлено {top_k} наиболее важных признаков для SHAP")

    # 1. Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shap_summary_{model_name}.png'), dpi=150)
    plt.close()
    logger.info(f"Summary plot сохранен в {output_dir}")

    # 2. Bar plot важности (средние абсолютные значения)
    shap_importance = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    importance_df.to_csv(os.path.join(output_dir, f'shap_feature_importance_{model_name}.csv'), index=False)
    logger.info("Важность признаков (SHAP) сохранена в CSV")

    # 3. Waterfall plots для первых 3 примеров
    for idx in range(min(3, len(X_test))):
        plt.figure()
        # base_value может быть скаляром или массивом – берем первое значение
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[0] if len(base) > 0 else 0
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=base,
                data=X_test.iloc[idx].values if hasattr(X_test, 'iloc') else X_test[idx],
                feature_names=feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_waterfall_{model_name}_sample{idx+1}.png'), dpi=150)
        plt.close()
        logger.info(f"Waterfall plot для примера {idx+1} сохранен")

    # 4. Dependence plots для топ-5 признаков
    top_features = importance_df.head(5)['feature'].tolist()
    for feat in top_features:
        try:
            idx_feat = feature_names.index(feat)
        except ValueError:
            logger.warning(f"  Признак {feat} не найден, пропускаем")
            continue
        plt.figure()
        shap.dependence_plot(idx_feat, shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_dependence_{model_name}_{feat}.png'), dpi=150)
        plt.close()
        logger.info(f"Dependence plot для признака '{feat}' сохранен")

    return importance_df


def lime_interpretation(model, X_train, X_test, feature_names, model_name, output_dir, top_k=None):
    """
    LIME: объяснение для нескольких примеров.
    Сохраняет графики и агрегированную важность.
    """
    logger.info("** LIME Analysis **")

    # Если данных много, берем подвыборку для обучения explainer
    sample_size = min(500, len(X_train))
    if len(X_train) > sample_size:
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
    else:
        X_train_sample = X_train

    # Если top_k задан, ограничиваем число признаков (для ускорения)
    if top_k and top_k < len(feature_names):
        num_features = top_k
    else:
        num_features = min(20, len(feature_names))

    # Создаем LIME explainer
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_sample.values if hasattr(X_train_sample, 'values') else X_train_sample,
            feature_names=feature_names,
            mode='regression',
            discretize_continuous=True,
            random_state=42
        )
        logger.info("LimeTabularExplainer создан")
    except Exception as e:
        logger.error(f"Ошибка при создании LIME explainer: {e}")
        return

    # Объясняем несколько примеров (первые 3)
    lime_importances = []
    for idx in range(min(3, len(X_test))):
        instance = X_test.iloc[idx].values if hasattr(X_test, 'iloc') else X_test[idx]
        exp = explainer.explain_instance(
            instance,
            model.predict,
            num_features=num_features
        )
        # Сохраняем график
        fig = exp.as_pyplot_figure()
        plt.title(f'LIME Explanation (sample {idx+1})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'lime_explanation_{model_name}_sample{idx+1}.png'), dpi=150)
        plt.close()
        logger.info(f"LIME объяснение для примера {idx+1} сохранено")

        # Собираем важность для агрегации
        for feature, weight in exp.as_list():
            lime_importances.append({'feature': feature, 'weight': abs(weight)})

    # Агрегированная важность LIME
    if lime_importances:
        lime_df = pd.DataFrame(lime_importances)
        lime_agg = lime_df.groupby('feature')['weight'].mean().reset_index()
        lime_agg.columns = ['feature', 'lime_importance']
        lime_agg = lime_agg.sort_values('lime_importance', ascending=False)
        lime_agg.to_csv(os.path.join(output_dir, f'lime_feature_importance_{model_name}.csv'), index=False)
        logger.info("Агрегированная важность LIME сохранена в CSV")
    else:
        logger.warning("Не удалось собрать важность LIME")


def run_interpretation(data_path, model_path, use_text=False, output_dir='reports/figures',
                       sample_size=1000, top_k=20):
    """
    Основная функция интерпретации.
    Параметры:
        data_path – путь к CSV с данными
        model_path – путь к сохраненной модели (.pkl)
        use_text – использовать ли текстовые эмбеддинги (True/False)
        output_dir – папка для сохранения графиков
        sample_size – максимальное число строк для SHAP/LIME (для ускорения)
        top_k – число признаков для отображения в графиках (для лучшей читаемости)
    """
    # Создаем папку для графиков
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем данные и модель
    X_train, X_test, y_train, y_test, model, feature_names, scaler, embedder = load_data_and_model(
        data_path, model_path, use_text, sample_size
    )

    # Имя модели для подписей
    model_name = type(model).__name__

    # SHAP
    shap_importance = shap_interpretation(
        model, X_train, X_test, feature_names, model_name, output_dir, top_k
    )

    # LIME
    lime_interpretation(
        model, X_train, X_test, feature_names, model_name, output_dir, top_k
    )

    # Сохраняем мета-информацию
    summary = {
        'model': model_name,
        'features_used': len(feature_names),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'use_text': use_text,
        'shap_top5': shap_importance.head(5)['feature'].tolist() if shap_importance is not None else []
    }
    with open(os.path.join(output_dir, 'interpretation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Сводка интерпретации сохранена в {output_dir}/interpretation_summary.json")


if __name__ == '__main__':
    # Параметры командной строки
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'lyrics_11k.csv'
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'best_model.pkl'
    use_text = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    output_dir = sys.argv[4] if len(sys.argv) > 4 else 'reports/figures'
    sample_size = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    top_k = int(sys.argv[6]) if len(sys.argv) > 6 else 20

    run_interpretation(data_path, model_path, use_text, output_dir, sample_size, top_k)

    # Примеры команд для запуска:
    # > python interpret.py lyrics_11k.csv best_model_numeric.pkl false reports/exp_numeric_figures 2000 20
    # > python interpret.py lyrics_11k.csv best_model_text.pkl true reports/exp_texts_figures 2000 20