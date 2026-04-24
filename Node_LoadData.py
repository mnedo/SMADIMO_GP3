import os
import json
import pandas as pd

from Node_Memory import set_pipeline_state



ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data(input_str):
    """
    Принимает json: file_paths (список путей к Excel-файлам)
    Возвращает json: status, dataset_path (путь к объединенному файлу), metadata_path (путь к метаданным), rows, cols, columns, duplicate_rows_after_concat

    Загружает один или несколько Excel-файлов, объединяет их по строкам
    и сохраняет общий датасет для следующих нод
    """
    try:
        if isinstance(input_str, str):
            data = json.loads(input_str)
        elif isinstance(input_str, dict):
            data = input_str
        elif isinstance(input_str, list):
            data = {"file_paths": input_str}
        else:
            return {
                "status": "error",
                "error": "Неподдерживаемый тип входа для load_data",
                "message": "load_data завершилась с ошибкой"
            }

        file_paths = data.get("file_paths", [])

        if not file_paths:
            return {
                "status": "error",
                "error": "Не передан список file_paths",
                "message": "load_data завершилась с ошибкой"
            }

        dfs = []

        for path in file_paths:
            if not os.path.exists(path):
                return {
                    "status": "error",
                    "error": f"Файл не найден: {path}",
                    "message": "load_data завершилась с ошибкой"
                }

            df_part = pd.read_excel(path)
            dfs.append(df_part)

        df = pd.concat(dfs, ignore_index=True)

        dataset_path = os.path.join(ARTIFACT_DIR, "loaded_dataset.xlsx")
        metadata_path = os.path.join(ARTIFACT_DIR, "load_metadata.json")

        df.to_excel(dataset_path, index=False)

        result = {
            "status": "ok",
            "error": None,
            "message": "Данные успешно загружены и объединены",
            "source_files_count": len(file_paths),
            "source_paths": file_paths,
            "dataset_path": dataset_path,
            "metadata_path": metadata_path,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": df.columns.tolist(),
            "duplicate_rows_after_concat": int(df.duplicated().sum())
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        set_pipeline_state(dataset_path=dataset_path)
        print('load_data завершилась успешно')
        return result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "load_data завершилась с ошибкой"
        }
