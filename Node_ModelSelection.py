import json
from langchain_core.messages import HumanMessage, SystemMessage

from Node_Memory import set_pipeline_state, get_pipeline_state


def model_selection(input_str: str, llm=None) -> dict:
    '''
    Принимает json: dataset_path (путь к файлу), context (контекст по датасету от предыдущих нод)
    Возвращает json: status, output_path (путь к тому же файлу для следующей ноды), recommended_models (список рекомендуемых моделей с обоснованиями), context (обновленный контекст по датасету для следующих нод)

    На основе характеристик задачи предлагает список релевантных моделей для обучения с обоснованием
    '''
    try:
        if isinstance(input_str, str):
            params = json.loads(input_str)
        elif isinstance(input_str, dict):
            params = input_str
        else:
            return {
                "status": "error",
                "error": "Неподдерживаемый тип входа для model_selection"
            }
        state = get_pipeline_state()
        path = params.get('dataset_path') or state.get('featured_dataset_path') or state.get('preprocessed_dataset_path') or state.get('dataset_path')
        if not path:
            return {"status": "error", "error": "dataset_path не найден ни в аргументах, ни в pipeline_state"}

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

        set_pipeline_state(recommended_models=models)

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
