import json
import os

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "gemini-2.5-flash"
with open("config.json", encoding="utf-8") as f:
    api_key = json.load(f)["llm"][MODEL_NAME]

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.3,
    google_api_key=api_key,
)

tools = []


# Ноды Role C
from Node_LoadData import load_data
tools.append(Tool(
    name="load_data",
    func=load_data,
    description="Загружает датасет из CSV-файла по переданному пути. "
                "Возвращает JSON со статусом, путём к сохранённому датафрейму, "
                "размерностью (rows, cols) и списком колонок."))

from Node_EDA import run_eda
tools.append(Tool(
    name="run_eda",
    func=run_eda,
    description="Проводит первичный анализ данных: типы колонок, пропуски, "
                "статистики, выявляет числовые/категориальные/текстовые признаки. "
                "Вызывать после load_data."))
# Ноды Role D
from Node_PreprocessDecision import preprocess_decision
tools.append(Tool(
    name="preprocess_decision",
    func=lambda input_data: preprocess_decision(input_data, llm),
    description="Анализирует EDA-отчёт и принимает решения по обработке: "
                "как заполнять пропуски, что делать с выбросами, нужна ли "
                "балансировка классов. Возвращает план действий в виде JSON."))

from Node_PreprocessExecution import preprocess_execution
tools.append(Tool(
    name="preprocess_execution",
    func=preprocess_execution,
    description="Выполняет план предобработки, сформированный preprocess_decision. "
                "Возвращает путь к очищенному датафрейму и список выполненных действий."))
# Ноды Role E
from Node_FeatureEngineering import feature_engineering
tools.append(Tool(
    name="feature_engineering",
    func=lambda input_data: feature_engineering(input_data, llm),
    description="Создаёт минимум 2 новых признака на основе существующих "
                "(отношения, бинаризация, извлечение из дат, агрегаты и т.д.). "
                "Возвращает обогащённый датафрейм и описание добавленных признаков."))

from Node_ModelSelection import model_selection
tools.append(Tool(
    name="model_selection",
    func=lambda input_data: model_selection(input_data, llm),
    description="На основе характеристик задачи (тип: классификация/регрессия, "
                "размер датасета, баланс классов) предлагает список релевантных "
                "моделей для обучения с обоснованием."))

from Node_TrainModels import train_models
tools.append(Tool(
    name="train_models",
    func=lambda input_data: train_models(input_data, llm),
    description="Обучает переданный список моделей на train-выборке, "
                "оценивает на test-выборке. Возвращает метрики по каждой модели "
                "и указывает лучшую."))

from Node_TuneHyperparams import tune_hyperparams
tools.append(Tool(
    name="tune_hyperparams",
    func=lambda input_data: tune_hyperparams(input_data, llm),
    description="Подбирает оптимальные гиперпараметры для выбранной модели "
                "с помощью Optuna. Возвращает лучшие параметры и улучшенную метрику. "
                "Вызывать после train_models для дополнительной оптимизации."))


# Ноды Role F
'''
from Node_Memory import save_best_model, load_previous_best, compare_with_previous
tools.append(Tool(
    name="save_best_model",
    func=save_best_model,
    description="Сохраняет лучшую модель в .pkl и метрики в JSON для "
                "использования при следующих запусках агента."
))
tools.append(Tool(
    name="load_previous_best",
    func=load_previous_best,
    description="Загружает метрики и параметры лучшей модели из предыдущих "
                "запусков агента. Возвращает None, если это первый запуск."
))
tools.append(Tool(
    name="compare_with_previous",
    func=compare_with_previous,
    description="Сравнивает текущие метрики с сохранёнными из предыдущего "
                "запуска. Возвращает выводы: улучшение/ухудшение и насколько."
))



'''

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

SYSTEM_PROMPT = """
Ты — агент для автоматизации ML-пайплайна (regression) по датасету Oskelly.
Цель: подготовить данные и обучить модели для предсказания колонки "Цена" по характеристикам товара.

Локальный файл для старта: dataset.xlsx (лежит рядом с Agent.py).

Правила:
- Всегда работай пошагово и вызывай инструменты по мере необходимости.
- Ничего не выдумывай про данные: сначала загрузи датасет, затем сделай EDA, затем предобработку.
- Если входные параметры для инструмента нужны структурированные — передавай их как JSON-строку.
""".strip()
agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


if __name__ == "__main__":
    dataset_path = os.path.abspath("dataset.xlsx")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Не найден dataset.xlsx по пути: {dataset_path}. "
            f"Положите файл рядом с Agent.py или измените путь в Agent.py."
        )

    user_input = (
        "Запусти полный pipeline.\n"
        f"Сначала вызови load_data с JSON: {{\"file_paths\": [\"{dataset_path}\"]}}.\n"
        "Потом сделай EDA, затем preprocess_decision, затем preprocess_execution, затем feature_engineering, "
        "затем model_selection, затем train_models и tune_hyperparams."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(result)