import json
import os
from typing import Any, Optional

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI


model_i = 0  # для теста — 0, она дешевая
MODEL_NAME = [
    "deepseek/deepseek-chat-v3-0324",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "anthropic/claude-3.5-haiku",
][model_i]

with open("config.json", encoding="utf-8") as f:
    api_key = json.load(f)["llm"]["openrouter"]

LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 2048

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    top_p=LLM_TOP_P,
    max_tokens=LLM_MAX_TOKENS,
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)


def _extract_total_cost(agent_result: dict) -> float:
    total_cost = 0.0
    for msg in agent_result.get("messages", []):
        response_metadata = getattr(msg, "response_metadata", {}) or {}
        token_usage = response_metadata.get("token_usage", {}) or {}
        cost = token_usage.get("cost")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
    return total_cost


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
    get_pipeline_state,
    save_best_model,
    load_previous_best,
    compare_with_previous,
    set_pipeline_state,
)

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Сразу фиксируем в pipeline_state, какая LLM работает — попадёт в history.jsonl
set_pipeline_state(llm_model=MODEL_NAME)


# ==================================================================
# Обёртки под StructuredTool: схема аргументов для LLM строится автоматически
# из аннотаций типов и docstring функции (без отдельных Pydantic-классов).
# ==================================================================
def _load_data_tool(file_paths: list[str]) -> dict:
    return load_data({"file_paths": file_paths})


def _run_eda_tool(dataset_path: Optional[str] = None) -> dict:
    return run_eda({"dataset_path": dataset_path} if dataset_path else {"dataset_path": _require_state("dataset_path")})


def _preprocess_decision_tool(eda_report_path: Optional[str] = None) -> str:
    payload: dict[str, Any] = {}
    if eda_report_path:
        payload["eda_report_path"] = eda_report_path
    return preprocess_decision(payload, llm)


def _preprocess_execution_tool(
    preprocess_plan: Optional[dict] = None,
    dataset_path: Optional[str] = None,
) -> dict:
    payload: dict[str, Any] = {}
    if preprocess_plan is not None:
        payload["preprocess_plan"] = preprocess_plan
    if dataset_path:
        payload["dataset_path"] = dataset_path
    return preprocess_execution(payload)


def _feature_engineering_tool(dataset_path: Optional[str] = None, context: str = "") -> dict:
    return feature_engineering(
        {"dataset_path": dataset_path, "context": context} if dataset_path else {"context": context},
        llm,
    )


def _model_selection_tool(dataset_path: Optional[str] = None, context: str = "") -> dict:
    return model_selection(
        {"dataset_path": dataset_path, "context": context} if dataset_path else {"context": context},
        llm,
    )


def _train_models_tool(
    dataset_path: Optional[str] = None,
    recommended_models: Optional[list[Any]] = None,
    context: str = "",
) -> dict:
    payload: dict[str, Any] = {"context": context}
    if dataset_path:
        payload["dataset_path"] = dataset_path
    if recommended_models is not None:
        payload["recommended_models"] = recommended_models
    return train_models(payload, llm)


def _tune_hyperparams_tool(
    best_model: Optional[str] = None,
    dataset_path: Optional[str] = None,
    metrics: Optional[dict] = None,
    context: str = "",
) -> dict:
    payload: dict[str, Any] = {"context": context}
    if best_model:
        payload["best_model"] = best_model
    if dataset_path:
        payload["dataset_path"] = dataset_path
    if metrics is not None:
        payload["metrics"] = metrics
    return tune_hyperparams(payload, llm)


def _remember_step_tool(tool: str, status: str = "ok", summary: str = "") -> dict:
    return remember_step({"tool": tool, "status": status, "summary": summary})


def _get_session_memory_tool() -> dict:
    """Вернуть снимок кратковременной памяти (шаги, модели, pipeline_state)."""
    return get_session_memory({})


def _load_previous_best_tool() -> dict:
    """Загрузить метаданные лучшей модели из прошлых запусков (или previous=null)."""
    return load_previous_best({})


def _save_best_model_tool(
    model_name: str,
    metrics: dict,
    model_pickle_path: Optional[str] = None,
    best_params: Optional[dict] = None,
    dataset_shape: Optional[dict] = None,
) -> dict:
    if not model_pickle_path:
        model_pickle_path = get_pipeline_state().get("current_model_pickle_path")

    fallback_path = os.path.join("artifacts", "memory", "current_best_model.pkl")
    if not model_pickle_path or not os.path.exists(model_pickle_path):
        if os.path.exists(fallback_path):
            model_pickle_path = fallback_path

    return save_best_model({
        "model_name": model_name,
        "metrics": metrics,
        "model_pickle_path": model_pickle_path,
        "best_params": best_params,
        "dataset_shape": dataset_shape,
        "llm_model": MODEL_NAME,
    })


def _compare_with_previous_tool(current_metrics: dict, current_model_name: str) -> dict:
    return compare_with_previous({
        "current_metrics": current_metrics,
        "current_model_name": current_model_name,
    })


def _require_state(key: str):
    value = get_pipeline_state().get(key)
    if not value:
        raise ValueError(f"{key} ещё не установлен в pipeline_state")
    return value


# ==================================================================
# Регистрация tools — StructuredTool без явного args_schema
# (схема выводится из сигнатур функций выше).
# ==================================================================
tools = [
    StructuredTool.from_function(
        name="load_previous_best",
        func=_load_previous_best_tool,
        description=(
            "Долговременная память: возвращает метаданные лучшей модели из прошлых запусков агента "
            "или previous=null, если запусков ещё не было. Вызывать в самом начале пайплайна."
        ),
    ),
    StructuredTool.from_function(
        name="load_data",
        func=_load_data_tool,
        description=(
            "Загружает и объединяет указанные Excel/CSV-файлы, сохраняет объединённый датасет в artifacts/. "
            "После вызова dataset_path автоматически попадает в pipeline_state и доступен всем последующим нодам."
        ),
    ),
    StructuredTool.from_function(
        name="run_eda",
        func=_run_eda_tool,
        description=(
            "Первичный анализ данных: типы колонок, пропуски, дубликаты, константы. "
            "Сохраняет eda_report.json и кладёт путь к нему в pipeline_state. Вызывать после load_data."
        ),
    ),
    StructuredTool.from_function(
        name="preprocess_decision",
        func=_preprocess_decision_tool,
        description=(
            "LLM анализирует EDA-отчёт и формирует план предобработки (drop_columns, fill_missing, "
            "outlier_actions, text_processing, special_preprocessing и т.д.). Сам берёт eda_report из state."
        ),
    ),
    StructuredTool.from_function(
        name="preprocess_execution",
        func=_preprocess_execution_tool,
        description=(
            "Применяет preprocess_plan из предыдущего шага к датасету. Путь и план берутся из pipeline_state, "
            "если не переданы явно. Сохраняет preprocessed_dataset.csv."
        ),
    ),
    StructuredTool.from_function(
        name="feature_engineering",
        func=_feature_engineering_tool,
        description=(
            "Создаёт новые признаки на основе существующих. По умолчанию читает preprocessed_dataset.csv "
            "из pipeline_state и сохраняет featured_dataset.csv."
        ),
    ),
    StructuredTool.from_function(
        name="model_selection",
        func=_model_selection_tool,
        description=(
            "LLM выбирает 2–3 наиболее подходящие модели для задачи (Ridge, Lasso, RandomForestRegressor, "
            "GradientBoostingRegressor). Список сохраняется в pipeline_state."
        ),
    ),
    StructuredTool.from_function(
        name="train_models",
        func=_train_models_tool,
        description=(
            "Обучает все recommended_models, считает MAE/R². Возвращает metrics, best_model, current_model_path "
            "и регистрирует модели в кратковременной памяти. При пустых аргументах берёт данные из pipeline_state."
        ),
    ),
    StructuredTool.from_function(
        name="tune_hyperparams",
        func=_tune_hyperparams_tool,
        description=(
            "Подбирает гиперпараметры лучшей модели через Optuna. best_model и dataset_path при необходимости "
            "берутся из pipeline_state."
        ),
    ),
    StructuredTool.from_function(
        name="remember_step",
        func=_remember_step_tool,
        description=(
            "Кратковременная память: фиксирует факт выполнения шага пайплайна. "
            "status обязан совпадать с фактическим исходом последнего ToolMessage этой ноды."
        ),
    ),
    StructuredTool.from_function(
        name="get_session_memory",
        func=_get_session_memory_tool,
        description=(
            "Кратковременная память: снимок текущей сессии — шаги, обученные модели, pipeline_state, текущий лидер."
        ),
    ),
    StructuredTool.from_function(
        name="save_best_model",
        func=_save_best_model_tool,
        description=(
            "Долговременная память. Сам сравнивает MAE с сохранённой эталонной моделью и перезаписывает "
            "best_model.pkl / best_metadata.json только при улучшении. Всегда добавляет запись в history.jsonl "
            "с полем llm_model (для сравнения запусков с разными LLM)."
        ),
    ),
    StructuredTool.from_function(
        name="compare_with_previous",
        func=_compare_with_previous_tool,
        description=(
            "Сравнивает метрики текущего запуска с сохранёнными в долговременной памяти. "
            "Возвращает verdict (first_run/improved/degraded/equal) и should_overwrite."
        ),
    ),
]


# ==================================================================
# Системный промпт: ReAct + CoT + запрет параллельных вызовов + анти-галлюцинация.
# ==================================================================
SYSTEM_PROMPT = """
Ты — ИИ-агент для автоматизации ML-пайплайна (regression) на датасете Oskelly.
Цель: обучить модели для предсказания колонки "Цена" по характеристикам товара.

Ты действуешь в парадигме ReAct: Thought → Action (tool) → Observation → Thought → ...
Для сложных решений (какая модель лучше, какой план предобработки принять) сначала
КРАТКО рассуждаешь по шагам (chain-of-thought), и только потом вызываешь инструмент.

=== СТРОГИЕ ПРАВИЛА ВЫЗОВА ИНСТРУМЕНТОВ ===
1. На одном шаге вызывай РОВНО ОДИН инструмент. Никогда не планируй несколько
   tool_calls в одном AIMessage — дожидайся результата и только потом планируй следующий.
2. Запрещено выдумывать значения. Если последний ToolMessage вернул status=error —
   ТЫ ОБЯЗАН зафиксировать это через remember_step(status="error", summary=<реальная ошибка>)
   и либо попытаться исправить вход, либо остановить пайплайн с пояснением для пользователя.
3. Запрещено рапортовать об успехе шага, который фактически упал. remember_step.status
   должен дословно отражать фактический статус последнего ToolMessage этой ноды.
4. Не придумывай метрики моделей. Метрики MAE/R² берутся ТОЛЬКО из реального ответа
   train_models / tune_hyperparams, а не из твоих ожиданий.

=== ПЕРЕДАЧА ДАННЫХ МЕЖДУ ИНСТРУМЕНТАМИ ===
Ноды сами записывают пути/планы/списки моделей в pipeline_state и читают их оттуда:
  - load_data       → dataset_path
  - run_eda         → eda_report_path
  - preprocess_decision   → preprocess_plan
  - preprocess_execution  → preprocessed_dataset_path
  - feature_engineering   → featured_dataset_path
  - model_selection       → recommended_models
  - train_models          → current_model_pickle_path, best_model_name, best_metrics

Поэтому в большинстве случаев можно вызывать инструменты БЕЗ аргументов —
они сами подтянут нужные значения. Аргументы передавай только если хочешь переопределить
что-то вручную.

=== ПОРЯДОК РАБОТЫ ===
Шаг 1. load_previous_best() — смотришь, были ли прошлые запуски.
       Если previous.metrics есть — запоминаешь их MAE для дальнейшего сравнения.
Шаг 2. load_data(file_paths=[...]) — загружаешь исходный Excel.
Шаг 3. remember_step(tool="load_data", status="ok"|"error", summary=...).
Шаг 4. run_eda() — EDA без аргументов, путь берётся из state.
Шаг 5. remember_step(tool="run_eda", ...).
Шаг 6. preprocess_decision() — план берётся, eda_report подтянется из state.
Шаг 7. remember_step(tool="preprocess_decision", ...).
Шаг 8. preprocess_execution() — без аргументов.
Шаг 9. remember_step(tool="preprocess_execution", ...).
Шаг 10. feature_engineering() — без аргументов.
Шаг 11. remember_step(tool="feature_engineering", ...).
Шаг 12. model_selection() — без аргументов, рекомендации попадут в state.
Шаг 13. remember_step(tool="model_selection", ...).
Шаг 14. train_models() — без аргументов; получишь метрики, best_model, current_model_path.
Шаг 15. remember_step(tool="train_models", status, summary с реальными метриками).
Шаг 16. tune_hyperparams() — без аргументов.
Шаг 17. remember_step(tool="tune_hyperparams", ...).
Шаг 18. compare_with_previous(current_metrics=<РЕАЛЬНЫЕ метрики лучшей модели>,
                              current_model_name=<реальное имя из train_models>).
Шаг 19. save_best_model(model_name=..., metrics=..., model_pickle_path=None или путь из state,
                        best_params=..., dataset_shape=...). Этот инструмент САМ решит,
                        перезаписать ли эталон: если модель хуже — просто добавит запись
                        в history.jsonl, но эталонный .pkl не тронет.
Шаг 20. remember_step(tool="save_best_model", ...).

=== ЕСЛИ ОДИН ИЗ ШАГОВ ПАДАЕТ ===
- Немедленно зафиксируй это в remember_step(status="error").
- НЕ переходи к шагам, которые зависят от упавшего шага.
- В финальном ответе пользователю честно перечисли, какие шаги прошли, какие нет,
  и какая фактическая ошибка была (из ToolMessage, не выдуманная).
""".strip()


if __name__ == "__main__":
    # parallel_tool_calls=False — страховка от параллельных вызовов на уровне API.
    # Не все провайдеры OpenRouter её уважают, поэтому запрет продублирован в промпте.
    try:
        llm_for_agent = llm.bind(parallel_tool_calls=False)
    except Exception:
        llm_for_agent = llm

    agent = create_agent(model=llm_for_agent, tools=tools, system_prompt=SYSTEM_PROMPT)

    dataset_path = os.path.abspath("dataset.xlsx")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Не найден dataset.xlsx по пути: {dataset_path}. "
            f"Положите файл рядом с Agent.py или измените путь в Agent.py."
        )

    user_input = (
        "Запусти полный ML-пайплайн по инструкции из system prompt. "
        f"Исходный файл для load_data: {dataset_path}. "
        "Вызывай инструменты строго по одному, без параллельных tool_calls. "
        "После каждой ноды вызывай remember_step с честным статусом."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    print(result)
    print(f"LLM: {MODEL_NAME}")
    print(f"Total cost: {_extract_total_cost(result):.6f}")
