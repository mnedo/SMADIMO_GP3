import json
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
key = open('api_key').readline()

def feature_engineering(input_str: str, llm=None) -> dict:
    '''
    Принимает json: dataset_path (путь к файлу), context (контекст по датасету от предыдущих нод)
    Возвращает json: status, output_path (путь к новому файлу для следующей ноды), new_features (список новых признаков), context (обновленный контекст по датасету для следующих нод)

    Создает новые признаки в датасете на основе существующих
    '''
    try:
        params = json.loads(input_str)
        path = params['dataset_path']
        df = pd.read_excel(path)

        context = params.get('context', '')

        if llm is None:
            llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.3, google_api_key=key)

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