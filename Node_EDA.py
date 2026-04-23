import os
import json
import pandas as pd



def run_eda(input_str):
    try:
        data = json.loads(input_str)
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

        df = pd.read_excel(dataset_path)

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

        eda_report = {
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
            "missing_top": missing_top
        }

        eda_report_path = os.path.join(ARTIFACT_DIR, "eda_report.json")

        with open(eda_report_path, "w", encoding="utf-8") as f:
            json.dump(eda_report, f, ensure_ascii=False, indent=2)

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
