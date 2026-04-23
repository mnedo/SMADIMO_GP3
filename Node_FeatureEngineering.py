import json
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage


def feature_engineering(input_str: str, llm=None) -> dict:
    '''
    Принимает json: dataset_path (путь к файлу), context (контекст по датасету от предыдущих нод)
    Возвращает json: status, output_path (путь к новому файлу для следующей ноды), new_features (список новых признаков), context (обновленный контекст по датасету для следующих нод)

    Создает новые признаки в датасете на основе существующих
    '''
    try:
        if isinstance(input_str, str):
            params = json.loads(input_str)
        elif isinstance(input_str, dict):
            params = input_str
        else:
            return {
                "status": "error",
                "error": "Неподдерживаемый тип входа для feature_engineering"
            }
        path = params['dataset_path']
        df = pd.read_excel(path)

        context = params.get('context', '')

        if llm is None:
            return {
                "status": "error",
                "error": "LLM не передан в feature_engineering"
            }

        response = llm.invoke([
            SystemMessage(
                content='Ты эксперт по feature engineering. Отвечай только готовым Python-кодом без пояснений и без ```python```.'),
            HumanMessage(
                content=f'{context}\n\nНапиши Python-код который создаёт минимум 3 новых признака в датафрейме df. Используй только колонки которые есть в датасете.')
        ])

        original_columns = list(df.columns)
        exec(response.content)
        new_columns = [col for col in df.columns if col not in original_columns]

        output_path = path.replace('.xlsx', '_fe.csv')
        df.to_csv(output_path, index=False)

        return {
            'status': 'ok',
            'new_features': new_columns,
            'output_path': output_path,
            'context': context + f'\n\nFeature Engineering создал новые признаки: {new_columns}'
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }