import os
import json
import pickle
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from Node_Memory import register_trained_models, set_pipeline_state, get_pipeline_state, MEMORY_DIR, record_llm_call

CURRENT_BEST_MODEL_PATH = os.path.join(MEMORY_DIR, "current_best_model.pkl")

def train_models(input_str: str, llm=None) -> dict:
    '''
    Принимает json: dataset_path (путь к файлу), recommended_models (список моделей от предыдущей ноды), context (контекст по датасету от предыдущих нод)
    Возвращает json: status, output_path (путь к тому же файлу для следующей ноды), metrics (метрики MAE и R2 по каждой модели), best_model (название лучшей модели), trained_models (объекты обученных моделей), context (обновленный контекст по датасету для следующих нод)

    Обучает переданный список моделей на train-выборке и оценивает на test-выборке, возвращает метрики по моделям и указывает лучшую модель
    '''
    response = None
    try:
        if isinstance(input_str, str):
            params = json.loads(input_str)
        elif isinstance(input_str, dict):
            params = input_str
        else:
            return {
                "status": "error",
                "error": "Неподдерживаемый тип входа для train_models"
            }
        state = get_pipeline_state()
        path = params.get('dataset_path') or state.get('featured_dataset_path') or state.get('preprocessed_dataset_path') or state.get('dataset_path')
        if not path:
            return {"status": "error", "error": "dataset_path не найден ни в аргументах, ни в pipeline_state"}

        context = params.get('context', '')

        if llm is None:
            return {
                "status": "error",
                "error": "LLM не передан в train_models"
            }

        recommended_models = params.get('recommended_models') or state.get('recommended_models')
        if not recommended_models:
            return {"status": "error", "error": "recommended_models не найдены ни в аргументах, ни в pipeline_state"}

        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)

        with open("prompts.json", encoding="utf-8") as f:
            prompts_data = json.load(f)
        USER_PROMPT = prompts_data[prompts_data["PROMPT_STYLE"]]["train_models"]
        USER_PROMPT += f"""

DataFrame `df` уже загружен (таргет — "Цена").
Список моделей для обучения: {recommended_models}.
Контекст: {context}

Используй только pandas и sklearn. Заполни переменные:
- metrics: dict вида {{"НазваниеМодели": {{"mae": ..., "r2": ...}}}}
- best_model: строка с названием лучшей модели по R2
- trained_models: dict с обученными объектами моделей по названиям

Только Python-код, без пояснений и без блока ```python.
""".strip()

        response = llm.invoke([{"role": "user", "content": USER_PROMPT}])

        local_vars = {'df': df}
        exec(response.content, local_vars)
        metrics = local_vars['metrics']
        best_model = local_vars['best_model']
        trained_models = local_vars['trained_models']

        with open(CURRENT_BEST_MODEL_PATH, 'wb') as f:
            pickle.dump(trained_models[best_model], f)

        register_trained_models(
            models_dict=trained_models,
            best_name=best_model,
            metrics=metrics
        )

        set_pipeline_state(
            current_model_pickle_path=CURRENT_BEST_MODEL_PATH,
            best_model_name=best_model,
            best_metrics=metrics.get(best_model),
        )

        print('train_models завершилась успешно')
        record_llm_call(response, success=True)
        return {
            'status': 'ok',
            'metrics': metrics,
            'best_model': best_model,
            'trained_models': trained_models,
            'current_model_path': CURRENT_BEST_MODEL_PATH,
            'context': context + f'\n\nTrainModels обучил модели. Метрики: {metrics}. Лучшая модель: {best_model}'
        }

    except Exception as e:
        if response is not None:
            record_llm_call(response, success=False)
        return {
            'status': 'error',
            'error': str(e)
        }