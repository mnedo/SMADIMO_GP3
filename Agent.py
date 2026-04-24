import json
import os

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from Node_LoadData import load_data
from Node_EDA import run_eda
from Node_PreprocessDecision import preprocess_decision
from Node_PreprocessExecution import preprocess_execution
from Node_FeatureEngineering import feature_engineering
from Node_ModelSelection import model_selection
from Node_TrainModels import train_models
from Node_TuneHyperparams import tune_hyperparams
from Node_Memory import remember_step, get_session_memory, save_best_model, load_previous_best, compare_with_previous, set_pipeline_state

model_i = 1
MODEL_NAME = [
    "deepseek/deepseek-chat-v3-0324",
    "openai/gpt-5-mini",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-pro-preview",
][model_i]

with open("config.json", encoding="utf-8") as f:
    api_key = json.load(f)["llm"]["openrouter"]

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.3,
    top_p=0.9,
    max_tokens=10_000,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

os.makedirs("artifacts", exist_ok=True)
set_pipeline_state(llm_model=MODEL_NAME)


def _extract_total_cost(agent_result: dict) -> float: # to-delete
    total_cost = 0.0
    for msg in agent_result.get("messages", []):
        response_metadata = getattr(msg, "response_metadata", {}) or {}
        token_usage = response_metadata.get("token_usage", {}) or {}
        cost = token_usage.get("cost")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
    return total_cost



def preprocess_decision_tool(input_data=None):
    return preprocess_decision({} if input_data is None else input_data, llm)


def feature_engineering_tool(input_str=None):
    return feature_engineering({} if input_str is None else input_str, llm)


def model_selection_tool(input_str=None):
    return model_selection({} if input_str is None else input_str, llm)


def train_models_tool(input_str=None):
    return train_models({} if input_str is None else input_str, llm)


def tune_hyperparams_tool(input_str=None):
    return tune_hyperparams({} if input_str is None else input_str, llm)


tools = []
tools.append(StructuredTool.from_function(
    name="load_previous_best",
    func=load_previous_best,
    description="Долговременная память. Аргумент input_str: {} или пустой dict.",
))
tools.append(StructuredTool.from_function(
    name="load_data",
    func=load_data,
    description='Загрузка данных. input_str: dict/JSON с ключом file_paths (список путей к файлам).',
))
tools.append(StructuredTool.from_function(
    name="run_eda",
    func=run_eda,
    description="EDA. input_str: dict/JSON с ключом dataset_path.",
))
tools.append(StructuredTool.from_function(
    name="preprocess_decision",
    func=preprocess_decision_tool,
    description="План предобработки (LLM). input_str: dict/JSON, часто {}.",
))
tools.append(StructuredTool.from_function(
    name="preprocess_execution",
    func=preprocess_execution,
    description=(
        "Выполнить план предобработки. Аргумент input_data: dict/JSON с ключами dataset_path (опционально, иначе из state), "
        "preprocess_plan — скопируй объект целиком из ответа preprocess_decision без упрощения. "
        "Поле drop_columns в preprocess_plan должно быть таким же, как у preprocess_decision: список объектов с ключами column и reason; "
        "не сокращай до списка имён колонок-строк."
    ),
))
tools.append(StructuredTool.from_function(
    name="feature_engineering",
    func=feature_engineering_tool,
    description="Новые признаки. input_str: dict/JSON или {}.",
))
tools.append(StructuredTool.from_function(
    name="model_selection",
    func=model_selection_tool,
    description="Выбор моделей. input_str: dict/JSON или {}.",
))
tools.append(StructuredTool.from_function(
    name="train_models",
    func=train_models_tool,
    description="Обучение. input_str: dict/JSON или {}.",
))
tools.append(StructuredTool.from_function(
    name="tune_hyperparams",
    func=tune_hyperparams_tool,
    description="Optuna. input_str: dict/JSON или {}.",
))
tools.append(StructuredTool.from_function(
    name="remember_step",
    func=remember_step,
    description='Шаг в память. input_str: dict/JSON с tool, status, summary.',
))
tools.append(StructuredTool.from_function(
    name="get_session_memory",
    func=get_session_memory,
    description="Снимок сессии. input_str: {} или пустой dict.",
))
tools.append(StructuredTool.from_function(
    name="save_best_model",
    func=save_best_model,
    description="Сохранить лучшую модель. input_str: dict/JSON по контракту Node_Memory.",
))
tools.append(StructuredTool.from_function(
    name="compare_with_previous",
    func=compare_with_previous,
    description="Сравнение с прошлым. input_str: dict/JSON с current_metrics, current_model_name.",
))


SYSTEM_PROMPT = """
Запрещено задавать вопросы. Выполни ML-пайплайн. Инструменты — функции нод; аргумент у большинства один: input_str (dict или JSON-строка).
Один tool за шаг. Метрики только из ответов train_models / tune_hyperparams.
""".strip()


if __name__ == "__main__":
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    dataset_path = os.path.abspath("dataset.xlsx")

    user_input = (
        f"""
        Выполни полный pipeline ML-инженера. Обучи модель для предсказания цены. Шаги к выполнению: 
        Вначале load_data: input_str = {{"file_paths": ["{dataset_path.replace(chr(92), "/")}"]}}.      
        Дальше run_eda, preprocess_decision, preprocess_execution, feature_engineering,
        model_selection, train_models, tune_hyperparams, compare_with_previous, save_best_model; remember_step после шагов."""
    )
    print(f"{MODEL_NAME} начала работу")
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(result)
    print(f"Total cost: {_extract_total_cost(result):.6f}")
