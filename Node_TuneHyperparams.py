import json
import os
import optuna
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from Node_Memory import set_pipeline_state, get_pipeline_state, record_llm_call


def tune_hyperparams(input_str: str, llm=None) -> dict:
    '''
    Принимает json: dataset_path (путь к файлу), best_model (название лучшей модели), metrics (метрики после обучения), context (контекст от предыдущих нод)
    Возвращает json: status, best_params (лучшие гиперпараметры), best_score (улучшенный MAE), context (обновленный контекст по датасету для следующих нод)

    Подбирает оптимальные гиперпараметры для лучшей модели с помощью Optuna
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
                "error": "Неподдерживаемый тип входа для tune_hyperparams"
            }
        state = get_pipeline_state()
        path = params.get('dataset_path') or state.get('featured_dataset_path') or state.get('preprocessed_dataset_path') or state.get('dataset_path')
        best_model_name = params.get('best_model') or state.get('best_model_name')
        context = params.get('context', '')

        if not path:
            return {"status": "error", "error": "dataset_path не найден ни в аргументах, ни в pipeline_state"}
        if not best_model_name:
            return {"status": "error", "error": "best_model не найден ни в аргументах, ни в pipeline_state"}

        if llm is None:
            return {
                "status": "error",
                "error": "LLM не передан в tune_hyperparams"
            }

        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
        X = df.drop(columns=['Цена']).select_dtypes(include='number')
        y = df['Цена']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with open("prompts.json", encoding="utf-8") as f:
            prompts_data = json.load(f)
        USER_PROMPT = prompts_data[prompts_data["PROMPT_STYLE"]]["tune_hyperparams"]
        USER_PROMPT += f"""

Данные уже разделены на X_train, X_test, y_train, y_test.
Модель для оптимизации: {best_model_name}.
Контекст: {context}

Заполни переменные:
- best_params: dict с лучшими гиперпараметрами
- best_score: float, лучший MAE на тесте

Для RandomForestRegressor параметр max_features может быть только int, float, sqrt, log2 или None (значение auto недопустимо).
Только Python-код, без пояснений и без блока ```python.
""".strip()

        response = llm.invoke([{"role": "user", "content": USER_PROMPT}])

        local_vars = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }
        exec(response.content, local_vars)
        best_params = local_vars['best_params']
        best_score = round(local_vars['best_score'], 4)

        set_pipeline_state(best_params=best_params)

        record_llm_call(response, success=True)
        return {
            'status': 'ok',
            'best_params': best_params,
            'best_score': best_score,
            'context': context + f'\n\nTuneHyperparams подобрал гиперпараметры для {best_model_name}: {best_params}. Лучший MAE: {best_score}'
        }

    except Exception as e:
        if response is not None:
            record_llm_call(response, success=False)
        return {
            'status': 'error',
            'error': str(e)
        }