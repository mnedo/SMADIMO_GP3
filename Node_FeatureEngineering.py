import os
import json
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from Node_Memory import set_pipeline_state, get_pipeline_state

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

        response = llm.invoke([
            SystemMessage(
                content=(
                    'Ты эксперт по feature engineering для задачи регрессии цены. '
                    'Отвечай только корректным Python-кодом, который работает с DataFrame df, без пояснений и без ```python```. '
                    'Цель — улучшить предсказательную способность признаков (R2), сохраняя обобщающую способность модели. '
                    'Выбирай преобразования самостоятельно, опираясь на структуру данных и EDA-контекст, а не на шаблонные действия. '
                    'Нельзя использовать колонку с ценой (price, цена, cost и эквиваленты) для генерации признаков в любом виде. '
                    'Работай устойчиво с пропусками, типами и редкими категориями; код не должен падать при отсутствии ожидаемой колонки. '
                    'Для текстовых и категориальных признаков выбирай уместные представления с учетом кардинальности и полезности сигнала. '
                    'Избегай избыточно большого числа шумных и разреженных признаков, предпочитай компактные и информативные. '
                    'Не удаляй существующие колонки, только добавляй новые.'
                )),
            HumanMessage(
                content=(
                    f'{context}\n\nEDA report (если есть): {json.dumps(eda_report, ensure_ascii=False) if eda_report else "not_provided"}\n\n'
                    'Используй доступный EDA-контекст и самостоятельно выбери разумные преобразования для создания новых признаков в df. '
                    'Сфокусируйся на качестве для регрессии, устойчивости к пропускам и аккуратной работе с текстовыми признаками. '
                    'Используй только колонки, которые реально есть в df. Не используй цену для построения признаков. '
                    'Сгенерируй не меньше 3 новых признаков.'
                ))
        ])

        original_columns = list(df.columns)
        print(response.content)
        exec(response.content)
        new_columns = [col for col in df.columns if col not in original_columns]

        output_path = os.path.join(ARTIFACT_DIR, "featured_dataset.csv")
        df.to_csv(output_path, index=False)

        set_pipeline_state(featured_dataset_path=output_path)
        print('feature_engineering завершилась успешно')
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