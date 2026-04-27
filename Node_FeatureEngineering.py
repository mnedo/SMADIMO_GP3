import os
import json
import pandas as pd

from Node_Memory import set_pipeline_state, get_pipeline_state, record_llm_call

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _read_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def _read_eda_report(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def feature_engineering(input_str: str, llm=None) -> dict:
    '''
    Принимает json: dataset_path (путь к файлу), context (контекст по датасету от предыдущих нод)
    Возвращает json: status, output_path (путь к новому файлу для следующей ноды), new_features (список новых признаков), context (обновленный контекст по датасету для следующих нод)

    Создает новые признаки в датасете на основе существующих
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
                "error": "Неподдерживаемый тип входа для feature_engineering"
            }
        path = params.get('dataset_path') or get_pipeline_state().get('preprocessed_dataset_path')
        if not path:
            return {"status": "error", "error": "dataset_path не найден ни в аргументах, ни в pipeline_state"}
        df = _read_dataset(path)

        context = params.get('context', '')
        state = get_pipeline_state()
        eda_report_path = params.get("eda_report_path") or state.get("eda_report_path")
        eda_report = None
        if eda_report_path and os.path.exists(eda_report_path):
            try:
                eda_report = _read_eda_report(eda_report_path)
            except Exception:
                eda_report = None

        if llm is None:
            return {
                "status": "error",
                "error": "LLM не передан в feature_engineering"
            }

        with open("prompts.json", encoding="utf-8") as f:
            prompts_data = json.load(f)
        USER_PROMPT = prompts_data[prompts_data["PROMPT_STYLE"]]["feature_engineering"]
        USER_PROMPT += f"""

DataFrame `df` уже загружен. Колонки и типы:
{json.dumps({col: str(df[col].dtype) for col in df.columns}, ensure_ascii=False, indent=2)}

Контекст: {context}

EDA-отчёт:
{json.dumps(eda_report, ensure_ascii=False, indent=2) if eda_report is not None else "не передан"}

Используй только pandas. Только Python-код, без пояснений и без блока ```python.
""".strip()

        response = llm.invoke([{"role": "user", "content": USER_PROMPT}])
        

        original_columns = list(df.columns)
        # print(response.content)
        local_vars = {'df': df, 'pd': pd}
        exec(response.content, local_vars)
        df = local_vars['df']
        df = df.fillna(0)
        new_columns = [col for col in df.columns if col not in original_columns]

        output_path = os.path.join(ARTIFACT_DIR, "featured_dataset.csv")
        df.to_csv(output_path, index=False)

        set_pipeline_state(featured_dataset_path=output_path)
        print('feature_engineering завершилась успешно')
        record_llm_call(response, success=True)
        return {
            'status': 'ok',
            'new_features': new_columns,
            'output_path': output_path,
            'context': context + f'\n\nFeature Engineering создал новые признаки: {new_columns}'
        }

    except Exception as e:
        if response is not None:
            record_llm_call(response, success=False)
        return {
            'status': 'error',
            'error': str(e)
        }