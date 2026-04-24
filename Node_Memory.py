import os
import json
import shutil
import uuid
from datetime import datetime

'''
Память ИИ-агента для auto-ML пайплайна.

Кратковременная память — словарь _SESSION_MEMORY на уровне модуля, живёт
в рамках одного запуска Python-процесса. В нём агент хранит историю
вызванных нод и список моделей, обученных в текущей сессии.

Долговременная память — файлы в artifacts/memory/:
    best_model.pkl        — лучшая модель за всю историю запусков
    best_metadata.json    — метрики, параметры и имя лучшей модели
    history.jsonl         — append-only лог всех запусков агента
'''

ARTIFACT_DIR = "artifacts"
MEMORY_DIR = os.path.join(ARTIFACT_DIR, "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MEMORY_DIR, "best_model.pkl")
BEST_METADATA_PATH = os.path.join(MEMORY_DIR, "best_metadata.json")
HISTORY_PATH = os.path.join(MEMORY_DIR, "history.jsonl")

_SESSION_MEMORY = {
    "session_id": str(uuid.uuid4()),
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "steps": [],
    "models_trained": [],
    "best_model_name": None,
    "best_metrics": None,
    "best_params": None
}


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
    Возвращает json: status, session_id, started_at, steps, models_trained, best_model_name, best_metrics

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
            "best_params": _SESSION_MEMORY["best_params"]
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


def save_best_model(input_str):
    '''
    Принимает json: model_name (имя модели), metrics (mae, r2 и т.д.), model_pickle_path (путь к текущему .pkl), best_params (опционально), dataset_shape (опционально)
    Возвращает json: status, saved_model_path, saved_metadata_path, message

    Сохраняет лучшую модель текущего запуска в долговременную память:
    копирует pickle в artifacts/memory/best_model.pkl, обновляет best_metadata.json
    и добавляет запись о запуске в history.jsonl.
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

        shutil.copyfile(model_pickle_path, BEST_MODEL_PATH)

        metadata = {
            "model_name": model_name,
            "metrics": metrics,
            "best_params": best_params,
            "dataset_shape": dataset_shape,
            "session_id": _SESSION_MEMORY["session_id"],
            "saved_at": datetime.now().isoformat(timespec="seconds")
        }

        with open(BEST_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False, default=str) + "\n")

        return {
            "status": "ok",
            "error": None,
            "message": f"Лучшая модель {model_name} сохранена в долговременной памяти",
            "saved_model_path": BEST_MODEL_PATH,
            "saved_metadata_path": BEST_METADATA_PATH
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

        current_mae = current_metrics.get("mae")
        current_r2 = current_metrics.get("r2")

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
        prev_mae = prev_metrics.get("mae")
        prev_r2 = prev_metrics.get("r2")

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
