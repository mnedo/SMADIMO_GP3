import json
import optuna
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

key = open('api_key').readline()

def tune_hyperparams(input_str: str, llm=None) -> dict:
    '''
    Принимает json: dataset_path (путь к файлу), best_model (название лучшей модели), metrics (метрики после обучения), context (контекст от предыдущих нод)
    Возвращает json: status, best_params (лучшие гиперпараметры), best_score (улучшенный MAE), context (обновленный контекст по датасету для следующих нод)

    Подбирает оптимальные гиперпараметры для лучшей модели с помощью Optuna
    '''
    try:
        params = json.loads(input_str)
        path = params['dataset_path']
        best_model_name = params['best_model']
        context = params.get('context', '')

        if llm is None:
            llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.3, google_api_key=key)

        df = pd.read_csv(path)
        X = df.drop(columns=['Цена']).select_dtypes(include='number')
        y = df['Цена']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        response = llm.invoke([
            SystemMessage(
                content='Ты эксперт по машинному обучению. Отвечай только готовым Python-кодом без пояснений и без ```python```. Используй optuna и sklearn. Для RandomForestRegressor параметр max_features может быть только: int, float, sqrt, log2 или None. Значение auto недопустимо.'),
            HumanMessage(
                content=f'{context}\n\nДанные уже разделены: X_train, X_test, y_train, y_test.\nМодель для оптимизации: {best_model_name}.\n\nНапиши код который:\n1. Определяет функцию objective для Optuna\n2. Создаёт study с direction="minimize"\n3. Запускает study.optimize() с n_trials=50\n4. Сохраняет лучшие параметры в best_params (dict)\n5. Сохраняет лучший MAE в best_score (float)')
        ])

        local_vars = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }
        exec(response.content, local_vars)
        best_params = local_vars['best_params']
        best_score = round(local_vars['best_score'], 4)

        return {
            'status': 'ok',
            'best_params': best_params,
            'best_score': best_score,
            'context': context + f'\n\nTuneHyperparams подобрал гиперпараметры для {best_model_name}: {best_params}. Лучший MAE: {best_score}'
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

test_input = json.dumps({
    'dataset_path': 'all_items800_fe_clean.csv',
    'best_model': 'RandomForestRegressor',
    'metrics': {'RandomForestRegressor': {'mae': 11718.5, 'r2': 0.802}},
    'context': 'Датасет товаров Oskelly. Задача: регрессия, предсказываем Цену. Лучшая модель: RandomForestRegressor.'
})

print(tune_hyperparams(test_input))