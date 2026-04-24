import os
import json
import shutil
import uuid
from datetime import datetime

'''
Память ИИ-агента для auto-ML пайплайна.

Кратковременная память:
  _SESSION_MEMORY — словарь на уровне модуля, живёт в рамках одного запуска
  Python-процесса. Хранит историю шагов, обученные модели, текущего лидера
  и pipeline_state (единый стор путей/артефактов, которым обмениваются ноды).

Долговременная память — файлы в artifacts/memory/:
    best_model.pkl        — лучшая модель за всю историю запусков
    best_metadata.json    — метрики, параметры и имя лучшей модели
    history.jsonl         — append-only лог всех запусков агента (с llm_model)
'''

ARTIFACT_DIR = "artifacts"
MEMORY_DIR = os.path.join(ARTIFACT_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MEMORY_DIR, "best_model.pkl")
BEST_METADATA_PATH = os.path.join(MEMORY_DIR, "best_metadata.json")
HISTORY_PATH = os.path.join(MEMORY_DIR, "history.jsonl")

_PIPELINE_STATE_DEFAULTS = {
    "dataset_path": None,            # исходный загруженный датасет (xlsx/csv)
    "eda_report_path": None,         # путь к eda_report.json
    "preprocess_plan": None,         # план из preprocess_decision
    "preprocessed_dataset_path": None,
    "featured_dataset_path": None,   # после feature_engineering
    "recommended_models": None,      # из model_selection
    "best_model_name": None,
    "best_metrics": None,
    "best_params": None,
    "current_model_pickle_path": None,
    "llm_model": None,               # какая LLM использовалась в этом запуске
}

_SESSION_MEMORY = {
    "session_id": str(uuid.uuid4()),
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "steps": [],
    "models_trained": [],
    "best_model_name": None,
    "best_metrics": None,
    "best_params": None,
    "pipeline_state": dict(_PIPELINE_STATE_DEFAULTS),
}


def set_pipeline_state(**dct):
    '''
    Служебная функция: ноды вызывают её, чтобы положить в session-state новые
    путь/артефакт/план. LLM в этот стор писать не может — только ноды.
    '''
    for key, value in dct.items():
        if key in _PIPELINE_STATE_DEFAULTS:
            _SESSION_MEMORY["pipeline_state"][key] = value


def get_pipeline_state():
    '''
    Возвращает копию текущего pipeline_state. Ноды используют это, чтобы
    подтянуть dataset_path / план и т.п. без помощи LLM.
    '''
    return dict(_SESSION_MEMORY["pipeline_state"])


def register_trained_models(models_dict, best_name, metrics, best_params=None):
    '''
    Служебная (не-tool) функция для Node_TrainModels / Node_TuneHyperparams.
    Записывает список обученных в текущей сессии моделей и лучшую из них
    в кратковременную память. Возвращает краткую сводку.
    '''
    trained = []
    for model_name, metric_values in metrics.items():
        trained.append({
            "name": model_name,
            "metrics": metric_values
        })

    _SESSION_MEMORY["models_trained"] = trained
    _SESSION_MEMORY["best_model_name"] = best_name
    _SESSION_MEMORY["best_metrics"] = metrics.get(best_name)
    if best_params is not None:
        _SESSION_MEMORY["best_params"] = best_params

    set_pipeline_state(
        best_model_name=best_name,
        best_metrics=metrics.get(best_name),
        best_params=best_params,
    )

    return {
        "session_id": _SESSION_MEMORY["session_id"],
        "models_trained_count": len(trained),
        "best_model_name": best_name
    }


def remember_step(input_str):
    '''
    Принимает json: tool (имя ноды), status (ok/error), summary (краткое описание результата)
    Возвращает json: status, session_id, steps_count, message

    Добавляет запись о выполненном шаге в кратковременную память агента.
    '''
    try:
        if isinstance(input_str, str):
            data = json.loads(input_str)
        elif isinstance(input_str, dict):
            data = input_str
        else:
            return {
                "status": "error",
                "error": "Неподдерживаемый тип входа для remember_step",
                "message": "remember_step завершилась с ошибкой"
            }

        tool_name = data.get("tool")
        if not tool_name:
            return {
                "status": "error",
                "error": "Не передан tool",
                "message": "remember_step завершилась с ошибкой"
            }

        step_record = {
            "tool": tool_name,
            "status": data.get("status", "ok"),
            "summary": data.get("summary", ""),
            "at": datetime.now().isoformat(timespec="seconds")
        }

        _SESSION_MEMORY["steps"].append(step_record)

        return {
            "status": "ok",
            "error": None,
            "message": f"Шаг {tool_name} сохранён в кратковременной памяти",
            "session_id": _SESSION_MEMORY["session_id"],
            "steps_count": len(_SESSION_MEMORY["steps"])
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "remember_step завершилась с ошибкой"
        }


def get_session_memory(input_str):
    '''
    Принимает json: пустой объект {} (аргументы не используются)
    Возвращает json: status, session_id, started_at, steps, models_trained, best_model_name, best_metrics, pipeline_state

    Возвращает снимок кратковременной памяти текущей сессии.
    Агент использует этот инструмент, чтобы вспомнить, какие модели
    он уже обучил и какие шаги выполнил.
    '''
    try:
        snapshot = {
            "session_id": _SESSION_MEMORY["session_id"],
            "started_at": _SESSION_MEMORY["started_at"],
            "steps": list(_SESSION_MEMORY["steps"]),
            "models_trained": list(_SESSION_MEMORY["models_trained"]),
            "best_model_name": _SESSION_MEMORY["best_model_name"],
            "best_metrics": _SESSION_MEMORY["best_metrics"],
            "best_params": _SESSION_MEMORY["best_params"],
            "pipeline_state": dict(_SESSION_MEMORY["pipeline_state"]),
        }

        return {
            "status": "ok",
            "error": None,
            "message": "Снимок кратковременной памяти текущей сессии",
            "session_memory": snapshot
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "get_session_memory завершилась с ошибкой"
        }


def _extract_mae(metrics) -> float | None:
    '''
    Аккуратно достаёт MAE из разных форматов:
      - {"mae": 12.3, "r2": 0.8}
      - {"MAE": 12.3}
      - {"ModelName": {"mae": 12.3, "r2": 0.8}}  -> None (нужен уже разрешённый словарь)
    '''
    if not isinstance(metrics, dict):
        return None
    for key in ("mae", "MAE", "Mae"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def save_best_model(input_str):
    '''
    Принимает json: model_name, metrics (mae/r2), model_pickle_path, best_params (optional),
                    dataset_shape (optional), llm_model (optional)
    Возвращает json: status, verdict, saved_model_path, saved_metadata_path,
                     previous_model_name, delta_mae, was_overwritten, message

    Самодостаточная долговременная память:
      - на первом запуске всегда сохраняет модель как эталон
      - на повторных запусках сравнивает MAE текущей модели с сохранённой
        и перезаписывает best_model.pkl / best_metadata.json только если
        текущий MAE строго меньше (модель лучше)
      - запись в history.jsonl добавляется всегда, включая llm_model и verdict
    '''
    try:
        if isinstance(input_str, str):
            data = json.loads(input_str)
        elif isinstance(input_str, dict):
            data = input_str
        else:
            return {
                "status": "error",
                "error": "Неподдерживаемый тип входа для save_best_model",
                "message": "save_best_model завершилась с ошибкой"
            }

        model_name = data.get("model_name")
        metrics = data.get("metrics")
        model_pickle_path = data.get("model_pickle_path")

        if not model_name or metrics is None or not model_pickle_path:
            return {
                "status": "error",
                "error": "Нужны поля model_name, metrics, model_pickle_path",
                "message": "save_best_model завершилась с ошибкой"
            }

        if not os.path.exists(model_pickle_path):
            return {
                "status": "error",
                "error": f"Файл модели не найден: {model_pickle_path}",
                "message": "save_best_model завершилась с ошибкой"
            }

        best_params = data.get("best_params")
        dataset_shape = data.get("dataset_shape")
        llm_model = data.get("llm_model") or _SESSION_MEMORY["pipeline_state"].get("llm_model")

        current_mae = _extract_mae(metrics)

        previous_metadata = None
        previous_mae = None
        previous_model_name = None
        if os.path.exists(BEST_METADATA_PATH):
            try:
                with open(BEST_METADATA_PATH, encoding="utf-8") as f:
                    previous_metadata = json.load(f)
                previous_mae = _extract_mae(previous_metadata.get("metrics") or {})
                previous_model_name = previous_metadata.get("model_name")
            except Exception:
                previous_metadata = None

        if previous_metadata is None:
            verdict = "first_run"
            should_overwrite = True
            delta_mae = None
        elif current_mae is None or previous_mae is None:
            verdict = "equal"
            should_overwrite = False
            delta_mae = None
        elif current_mae < previous_mae:
            verdict = "improved"
            should_overwrite = True
            delta_mae = round(current_mae - previous_mae, 4)
        elif current_mae > previous_mae:
            verdict = "degraded"
            should_overwrite = False
            delta_mae = round(current_mae - previous_mae, 4)
        else:
            verdict = "equal"
            should_overwrite = False
            delta_mae = 0.0

        metadata = {
            "model_name": model_name,
            "metrics": metrics,
            "best_params": best_params,
            "dataset_shape": dataset_shape,
            "llm_model": llm_model,
            "session_id": _SESSION_MEMORY["session_id"],
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "verdict": verdict,
        }

        saved_model_path = BEST_MODEL_PATH if should_overwrite else (
            previous_metadata.get("model_pickle_path") if previous_metadata else None
        )

        if should_overwrite:
            shutil.copyfile(model_pickle_path, BEST_MODEL_PATH)
            metadata["model_pickle_path"] = BEST_MODEL_PATH
            with open(BEST_METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            saved_metadata_path = BEST_METADATA_PATH
            saved_model_path = BEST_MODEL_PATH
        else:
            saved_metadata_path = BEST_METADATA_PATH

        history_record = dict(metadata)
        history_record["was_overwritten"] = should_overwrite
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(history_record, ensure_ascii=False, default=str) + "\n")

        if verdict == "first_run":
            msg = f"Первый запуск: модель {model_name} сохранена как эталон"
        elif verdict == "improved":
            msg = f"Модель {model_name} лучше прошлой на {abs(delta_mae)} по MAE — best_model перезаписан"
        elif verdict == "degraded":
            msg = (
                f"Модель {model_name} хуже прошлой на {delta_mae} по MAE — "
                f"best_model НЕ перезаписан, запись добавлена только в history"
            )
        else:
            msg = f"MAE модели {model_name} не отличается от прошлой — best_model НЕ перезаписан"

        return {
            "status": "ok",
            "error": None,
            "verdict": verdict,
            "was_overwritten": should_overwrite,
            "delta_mae": delta_mae,
            "previous_model_name": previous_model_name,
            "saved_model_path": saved_model_path,
            "saved_metadata_path": saved_metadata_path,
            "message": msg,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "save_best_model завершилась с ошибкой"
        }


def load_previous_best(input_str):
    '''
    Принимает json: пустой объект {} (аргументы не используются)
    Возвращает json: status, previous (метаданные прошлого лучшего запуска или null), message

    Загружает метаданные лучшей модели из предыдущих запусков агента.
    Если сохранённого запуска ещё нет, возвращает previous=null.
    '''
    try:
        if not os.path.exists(BEST_METADATA_PATH):
            return {
                "status": "ok",
                "error": None,
                "message": "Это первый запуск агента, предыдущих результатов нет",
                "previous": None
            }

        with open(BEST_METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)

        return {
            "status": "ok",
            "error": None,
            "message": f"Загружены метрики предыдущего лучшего запуска: {metadata.get('model_name')}",
            "previous": metadata
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "load_previous_best завершилась с ошибкой"
        }


def compare_with_previous(input_str):
    '''
    Принимает json: current_metrics (mae, r2), current_model_name (имя текущей лучшей модели)
    Возвращает json: status, verdict, delta_mae, delta_r2, should_overwrite, previous_model_name, message

    Сравнивает текущие метрики с сохранёнными в долговременной памяти.
    Verdict: first_run | improved | degraded | equal.
    should_overwrite=True если MAE уменьшился (или это первый запуск).
    '''
    try:
        if isinstance(input_str, str):
            data = json.loads(input_str)
        elif isinstance(input_str, dict):
            data = input_str
        else:
            return {
                "status": "error",
                "error": "Неподдерживаемый тип входа для compare_with_previous",
                "message": "compare_with_previous завершилась с ошибкой"
            }

        current_metrics = data.get("current_metrics")
        current_model_name = data.get("current_model_name")

        if current_metrics is None or not current_model_name:
            return {
                "status": "error",
                "error": "Нужны поля current_metrics и current_model_name",
                "message": "compare_with_previous завершилась с ошибкой"
            }

        current_mae = _extract_mae(current_metrics)
        current_r2 = current_metrics.get("r2") if isinstance(current_metrics, dict) else None

        if not os.path.exists(BEST_METADATA_PATH):
            return {
                "status": "ok",
                "error": None,
                "verdict": "first_run",
                "delta_mae": None,
                "delta_r2": None,
                "should_overwrite": True,
                "previous_model_name": None,
                "message": "Предыдущего запуска нет — текущий результат станет эталоном"
            }

        with open(BEST_METADATA_PATH, encoding="utf-8") as f:
            previous = json.load(f)

        prev_metrics = previous.get("metrics", {})
        prev_mae = _extract_mae(prev_metrics)
        prev_r2 = prev_metrics.get("r2") if isinstance(prev_metrics, dict) else None

        delta_mae = None
        delta_r2 = None
        if current_mae is not None and prev_mae is not None:
            delta_mae = round(current_mae - prev_mae, 4)
        if current_r2 is not None and prev_r2 is not None:
            delta_r2 = round(current_r2 - prev_r2, 4)

        if delta_mae is None:
            verdict = "equal"
            should_overwrite = False
        elif delta_mae < 0:
            verdict = "improved"
            should_overwrite = True
        elif delta_mae > 0:
            verdict = "degraded"
            should_overwrite = False
        else:
            verdict = "equal"
            should_overwrite = False

        if verdict == "improved":
            msg = f"Текущая модель {current_model_name} лучше: MAE улучшился на {abs(delta_mae)}"
        elif verdict == "degraded":
            msg = f"Текущая модель {current_model_name} хуже предыдущей на {delta_mae} по MAE"
        else:
            msg = f"Текущая модель {current_model_name} по MAE не отличается от предыдущей"

        return {
            "status": "ok",
            "error": None,
            "verdict": verdict,
            "delta_mae": delta_mae,
            "delta_r2": delta_r2,
            "should_overwrite": should_overwrite,
            "previous_model_name": previous.get("model_name"),
            "message": msg
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "compare_with_previous завершилась с ошибкой"
        }
