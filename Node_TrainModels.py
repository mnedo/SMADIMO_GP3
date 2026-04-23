import json
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_models(input_str: str, llm=None) -> dict:
    '''
    Принимает json: dataset_path (путь к файлу), recommended_models (список моделей от предыдущей ноды), context (контекст по датасету от предыдущих нод)
    Возвращает json: status, output_path (путь к тому же файлу для следующей ноды), metrics (метрики MAE и R2 по каждой модели), best_model (название лучшей модели), trained_models (объекты обученных моделей), context (обновленный контекст по датасету для следующих нод)

    Обучает переданный список моделей на train-выборке и оценивает на test-выборке, возвращает метрики по моделям и указывает лучшую модель
    '''
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
        path = params['dataset_path']

        context = params.get('context', '')

        if llm is None:
            return {
                "status": "error",
                "error": "LLM не передан в train_models"
            }

        recommended_models = params['recommended_models']

        df = pd.read_csv(path)

        response = llm.invoke([
            SystemMessage(
                content='Ты эксперт по машинному обучению. Отвечай только готовым Python-кодом без пояснений и без ```python```. Используй только библиотеки: pandas, sklearn. Результаты сохраняй в переменные metrics (dict) и best_model (str).'),
            HumanMessage(
                content=f'{context}\n\nДатафрейм уже загружен в переменную df. Список моделей для обучения: {recommended_models}.\n\nНапиши код который:\n1. Делит df на X (числовые колонки кроме Цена) и y (Цена)\n2. Делает train_test_split\n3. Обучает каждую модель из списка\n4. Считает MAE и R2 для каждой\n5. Сохраняет результаты в metrics = {{"НазваниеМодели": {{"mae": ..., "r2": ...}}}}\n6. Сохраняет название лучшей модели по R2 в best_model\n7. Сохраняй каждый обученный объект модели в trained_models[НазваниеМодели]')
        ])

        local_vars = {'df': df}
        exec(response.content, local_vars)
        metrics = local_vars['metrics']
        best_model = local_vars['best_model']

        return {
            'status': 'ok',
            'metrics': metrics,
            'best_model': best_model,
            'trained_models': local_vars['trained_models'],
            'context': context + f'\n\nTrainModels обучил модели. Метрики: {metrics}. Лучшая модель: {best_model}'
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }