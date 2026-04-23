import json
from langchain_core.messages import HumanMessage, SystemMessage


def model_selection(input_str: str, llm=None) -> dict:
    '''
    Принимает json: dataset_path (путь к файлу), context (контекст по датасету от предыдущих нод)
    Возвращает json: status, output_path (путь к тому же файлу для следующей ноды), recommended_models (список рекомендуемых моделей с обоснованиями), context (обновленный контекст по датасету для следующих нод)

    На основе характеристик задачи предлагает список релевантных моделей для обучения с обоснованием
    '''
    try:
        params = json.loads(input_str)
        path = params['dataset_path']

        context = params.get('context', '')

        if llm is None:
            return {
                "status": "error",
                "error": "LLM не передан в model_selection"
            }

        response = llm.invoke([
            SystemMessage(
                content='Ты эксперт в области машинного обучения. Отвечай только валидным JSON без пояснений и без ```json```. Формат ответа: [{"model": "НазваниеМодели", "reason": "обоснование"}]. Выбирай только из этих моделей: Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor.'),
            HumanMessage(
                content=f'{context}\n\nВыбери 2-3 наиболее подходящие модели для этой задачи с обоснованием.')
        ])
        models = json.loads(response.content)

        return {
            'status': 'ok',
            'output_path': path,
            'recommended_models': models,
            'context': context + f'\n\nModelSelection определил список релевантных моделей для обучения: {models}'
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }
