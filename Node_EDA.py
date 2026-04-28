import os
import json
import pandas as pd

from Node_Memory import set_pipeline_state

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _read_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def run_eda(input_str):
    """
    Принимает json: dataset_path (путь к объединенному Excel-файлу)
    Возвращает json: status, eda_report_path, eda_report, feature_type_counts

    Выполняет базовый EDA по датасету: определяет числовые, категориальные и
    текстовые колонки, считает пропуски, дубликаты и константные признаки.
    """
    try:
        if isinstance(input_str, str):
            data = json.loads(input_str)
        elif isinstance(input_str, dict):
            data = input_str
        else:
            return {
                "status": "error",
                "error": "Неподдерживаемый тип входа для run_eda",
                "message": "run_eda завершилась с ошибкой"
            }
        dataset_path = data.get("dataset_path")

        if not dataset_path:
            return {
                "status": "error",
                "error": "Не передан dataset_path",
                "message": "run_eda завершилась с ошибкой"
            }

        if not os.path.exists(dataset_path):
            return {
                "status": "error",
                "error": f"Файл не найден: {dataset_path}",
                "message": "run_eda завершилась с ошибкой"
            }

        df = _read_dataset(dataset_path)

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        object_columns = df.select_dtypes(include=["object"]).columns.tolist()

        text_columns = []
        categorical_columns = []

        for col in object_columns:
            if col == "Описание":
                text_columns.append(col)
            else:
                categorical_columns.append(col)

        missing_top = []
        missing_counts = df.isna().sum()
        missing_pct = (df.isna().mean() * 100).round(2)

        for col in df.columns:
            if missing_counts[col] > 0:
                missing_top.append({
                    "column": col,
                    "missing_count": int(missing_counts[col]),
                    "missing_percent": float(missing_pct[col])
                })

        missing_top = sorted(
            missing_top,
            key=lambda x: x["missing_percent"],
            reverse=True
        )[:15]

        constant_columns = [
            col for col in df.columns
            if df[col].nunique(dropna=False) <= 1
        ]

        numeric_distributions = {}
        for col in numeric_columns:
            series = df[col].dropna()
            if series.empty:
                continue
            numeric_distributions[col] = {
                "count_non_null": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std()) if series.shape[0] > 1 else 0.0,
                "min": float(series.min()),
                "q01": float(series.quantile(0.01)),
                "q25": float(series.quantile(0.25)),
                "median": float(series.quantile(0.5)),
                "q75": float(series.quantile(0.75)),
                "q99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "skew": float(series.skew()) if series.shape[0] > 2 else 0.0
            }

        categorical_profiles = {}
        for col in categorical_columns:
            non_null = df[col].dropna().astype(str).str.strip()
            unique_count = int(non_null.nunique())
            top_values = non_null.value_counts().head(10)
            categorical_profiles[col] = {
                "unique_count": unique_count,
                "is_high_cardinality": unique_count > 50,
                "top_values": [
                    {"value": str(idx), "count": int(cnt)}
                    for idx, cnt in top_values.items()
                ]
            }

        text_profiles = {}
        for col in text_columns:
            text_series = df[col].dropna().astype(str)
            if text_series.empty:
                continue
            token_lens = text_series.str.split().str.len()
            char_lens = text_series.str.len()
            text_profiles[col] = {
                "count_non_null": int(text_series.shape[0]),
                "avg_chars": float(char_lens.mean()),
                "avg_words": float(token_lens.mean()),
                "q95_chars": float(char_lens.quantile(0.95))
            }

        dataset_description = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_cells_total": int(df.isna().sum().sum()),
            "missing_cells_percent": float((df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] and df.shape[1] else 0.0),
            "duplicate_rows": int(df.duplicated().sum()),
            "constant_columns_count": int(len(constant_columns))
        }

        eda_report = {
            "dataset_description": dataset_description,
            "dataset_shape": {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1])
            },
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "text_columns": text_columns,
            "duplicate_rows": int(df.duplicated().sum()),
            "constant_columns": constant_columns,
            "missing_top": missing_top,
            "numeric_distributions": numeric_distributions,
            "categorical_profiles": categorical_profiles,
            "text_profiles": text_profiles
        }

        eda_report_path = os.path.join(ARTIFACT_DIR, "eda_report.json")

        with open(eda_report_path, "w", encoding="utf-8") as f:
            json.dump(eda_report, f, ensure_ascii=False, indent=2)

        set_pipeline_state(eda_report_path=eda_report_path, dataset_path=dataset_path)
        print('run_eda завершилась успешно')
        return {
            "status": "ok",
            "error": None,
            "message": "EDA успешно выполнен",
            "eda_report_path": eda_report_path,
            "eda_report": eda_report,
            "feature_type_counts": {
                "numeric_count": len(numeric_columns),
                "categorical_count": len(categorical_columns),
                "text_count": len(text_columns)
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "run_eda завершилась с ошибкой"
        }
