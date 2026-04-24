import json
import os

from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

model_i = 0 # для теста - 0, она дешевая
MODEL_NAME = [
    "deepseek/deepseek-chat-v3-0324",
    "anthropic/claude-opus-4.7", # based on SWE-Benchmark-2026
    "google/gemini-3.1-pro-preview", # based on SWE-Benchmark-2026
    "moonshotai/kimi-k2-thinking" # based on SWE-Benchmark-2026
][model_i]
with open("config.json", encoding="utf-8") as f:
    api_key = json.load(f)["llm"]['openrouter']


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.3,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
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
from Node_Memory import (
    remember_step,
    get_session_memory,
    save_best_model,
    load_previous_best,
    compare_with_previous,
)
tools.append(Tool(
    name="remember_step",
    func=remember_step,
    description="Кратковременная память: записывает в текущую сессию факт выполнения "
                "очередного шага пайплайна. Передавай JSON с полями tool (имя ноды), "
                "status (ok/error), summary (краткое описание результата)."))
tools.append(Tool(
    name="get_session_memory",
    func=get_session_memory,
    description="Кратковременная память: возвращает снимок текущей сессии — список "
                "выполненных шагов, обученные модели и текущий лидер. Вызывай, чтобы "
                "вспомнить, какие модели уже обучены и что делалось. На вход — {}"))
tools.append(Tool(
    name="save_best_model",
    func=save_best_model,
    description="Долговременная память: сохраняет лучшую модель в .pkl и метрики в JSON "
                "для использования при следующих запусках агента. Принимает JSON "
                "{model_name, metrics, model_pickle_path, best_params?, dataset_shape?}."))
tools.append(Tool(
    name="load_previous_best",
    func=load_previous_best,
    description="Долговременная память: загружает метрики и имя лучшей модели из "
                "предыдущих запусков агента. Возвращает previous=null, если это первый "
                "запуск. Вызывай в самом начале пайплайна. На вход — {}"))
tools.append(Tool(
    name="compare_with_previous",
    func=compare_with_previous,
    description="Долговременная память: сравнивает текущие метрики с сохранёнными из "
                "предыдущего лучшего запуска. Возвращает verdict (improved/degraded/"
                "equal/first_run) и should_overwrite. Принимает JSON "
                "{current_metrics, current_model_name}."))

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

Работа с памятью (обязательно):
- В самом начале пайплайна вызови load_previous_best с {}. Если previous=null — это
  первый запуск. Если previous есть — запомни его метрики (особенно mae), они пригодятся
  для сравнения в конце.
- После каждой ключевой ноды (load_data, run_eda, preprocess_execution, feature_engineering,
  train_models, tune_hyperparams) вызывай remember_step, передавая tool, status и короткий
  summary. Это кратковременная память текущей сессии.
- Если нужно вспомнить, какие модели уже обучены в этой сессии — вызывай get_session_memory
  с {}.
- После train_models возьми из его результата поле current_model_path — это путь к pickle
  лучшей модели текущего запуска. Он понадобится для save_best_model.
- После tune_hyperparams (или после train_models, если тюнинг пропущен) вызови
  compare_with_previous, передав JSON {"current_metrics": <метрики лучшей модели>,
  "current_model_name": <имя лучшей модели>}.
- В самом конце вызови save_best_model с JSON {"model_name": <имя>, "metrics": <метрики>,
  "model_pickle_path": <current_model_path>, "best_params": <params или null>,
  "dataset_shape": {"rows": ..., "cols": ...}}. На первом запуске — вызывай всегда.
  На последующих — ориентируйся на verdict от compare_with_previous: при improved/first_run
  сохраняем обязательно, при degraded/equal можно пропустить.
""".strip()
if __name__ == "__main__":
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

    dataset_path = os.path.abspath("dataset.xlsx")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Не найден dataset.xlsx по пути: {dataset_path}. "
            f"Положите файл рядом с Agent.py или измените путь в Agent.py."
        )

    user_input = (
        "Запусти полный pipeline.\n"
        "Сначала вызови load_previous_best с {}, чтобы узнать результаты прошлых запусков.\n"
        f"Затем вызови load_data с JSON: {{\"file_paths\": [\"{dataset_path}\"]}}.\n"
        "Потом сделай EDA, затем preprocess_decision, затем preprocess_execution, затем feature_engineering, "
        "затем model_selection, затем train_models и tune_hyperparams. "
        "После каждого шага вызывай remember_step.\n"
        "В конце вызови compare_with_previous с текущими метриками лучшей модели, "
        "а затем save_best_model, передав model_pickle_path из current_model_path результата train_models."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(result)