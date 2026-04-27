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
from Node_Memory import (
    remember_step,
    get_session_memory,
    save_best_model,
    load_previous_best,
    compare_with_previous,
    set_pipeline_state,
    record_llm_call,
    get_llm_usage,
)

model_i = 0
MODEL_NAME = [
    "deepseek/deepseek-chat-v3-0324",
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-pro-preview", # sota for agent at ok-cost
][model_i]

prompt_i = 1
PROMPT_STYLES = [
    "Baseline",
    "Few-shot",
    "CoT",
    "Role",
    "Contrastive",
]
PROMPT_STYLE = PROMPT_STYLES[prompt_i]

with open("prompts.json", encoding="utf-8") as f:
    prompts_data = json.load(f)
prompts_data["PROMPT_STYLE"] = PROMPT_STYLE
with open("prompts.json", "w", encoding="utf-8") as f:
    json.dump(prompts_data, f, ensure_ascii=False, indent=2)


with open("config.json", encoding="utf-8") as f:
    api_key = json.load(f)["llm"]["openrouter"]

LLM_PARAMS = {
    "model": MODEL_NAME,
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 50_000,
}

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    **LLM_PARAMS,
)

os.makedirs("artifacts", exist_ok=True)
set_pipeline_state(
    llm_model=MODEL_NAME,
    prompt_style=PROMPT_STYLE,
    llm_params=LLM_PARAMS,
)


def _register_agent_llm_usage(agent_result: dict) -> None:
    """
    Регистрирует все LLM-вызовы оркестратора в session-памяти после agent.invoke.
    Тулзы регистрируют свои внутренние вызовы сами через record_llm_call.
    """
    for msg in agent_result.get("messages", []):
        response_metadata = getattr(msg, "response_metadata", {}) or {}
        if response_metadata.get("token_usage"):
            record_llm_call(msg, success=True)



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
Запрещено задавать вопросы. Инструменты — функции нод; аргумент у большинства один: input_str (dict или JSON-строка). Один tool за шаг.
""".strip()


if __name__ == "__main__":
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

    data_dir = os.path.join("data", "all_items")
    file_paths = [
        os.path.join(data_dir, name).replace("\\", "/")
        for name in sorted(os.listdir(data_dir))
        if name.lower().endswith(".xlsx") and name.startswith("all_items")
    ]
    if not file_paths:
        raise FileNotFoundError(f"В папке {data_dir} не найдено файлов all_items*.xlsx")

    with open("prompts.json", encoding="utf-8") as f:
        data = json.load(f)
        USER_PROMPT = data[data["PROMPT_STYLE"]]["agent"]
    USER_PROMPT += f"\nПуть к датасету: {json.dumps({'file_paths': file_paths}, ensure_ascii=False)}"
    
    USER_PROMPT += """\nКонтекст задачи:
    - Бизнес-заказчик: Oskelly
    - Задача: подобрать релевантную цену для вещи по её характеристикам
    - Тип ML-задачи: regression
    - Целевая колонка: Цена
    
    Особенности предметной области:
    - Колонка "Размер" может содержать значения в смешанных системах, например: FR 40, IT 40, EU 40, INT M, INT XS, JEANS 31.
    - Колонка "Скидка" может содержать значения вида "-11%".
    - Если в колонке "Скидка" есть пропуски, считай это отсутствием скидки
""".strip()
    print(f"{MODEL_NAME} начала работу (prompt_style={PROMPT_STYLE})")
    result = agent.invoke({"messages": [{"role": "user", "content": USER_PROMPT}]})
    _register_agent_llm_usage(result)
    print(result)
    usage = get_llm_usage()
    print(
        f"LLM usage: calls={usage['total_calls']}, "
        f"failed={usage['failed_calls']} ({usage['error_rate']:.2%}), "
        f"tokens={usage['tokens']['total']}, total_cost={usage['total_cost']:.6f}"
    )
