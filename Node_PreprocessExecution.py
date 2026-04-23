import os
import json
import numpy as np
import pandas as pd

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def preprocess_execution(input_data):
      """
    Принимает json: dataset_path (путь к файлу датасета), preprocess_plan (план предобработки от предыдущей ноды)
    Возвращает json: status, preprocessed_dataset_path, final_shape, applied_actions

    Применяет к датасету план предобработки: удаление колонок и дубликатов, обработку пропусков, специальные преобразования, очистку текстовых признаков.
    Сохраняет предобработанный датасет в файл и возвращает путь к нему вместе со списком выполненных действий.
    """
    try:
        data = json.loads(input_data) if isinstance(input_data, str) else input_data
        dataset_path = data["dataset_path"]
        preprocess_plan = data["preprocess_plan"]

        if isinstance(preprocess_plan, str):
            preprocess_plan = json.loads(preprocess_plan)

        df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_excel(dataset_path)

        a = []
        target_col = "Цена"
        if target_col in df.columns:
            before = len(df)
            df = df.dropna(subset=[target_col])
            a.append(f"Удалено строк с пустой ценой: {before - len(df)}")

        cols_to_drop = [
            item["column"]
            for item in preprocess_plan.get("drop_columns", [])
        ]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            a.append(f"Удалены колонки: {cols_to_drop}")

        if preprocess_plan.get("drop_duplicates"):
            before = len(df)
            df = df.drop_duplicates()
            a.append(f"Удалено дубликатов: {before - len(df)}")

        for i, j in preprocess_plan.get("special_preprocessing", {}).items():
            if i not in df.columns:
                continue
            if j == "percent_to_float":
                df[i] = df[i].str.replace("-", "").str.replace("%", "")
                df[i] = pd.to_numeric(df[i], errors="coerce")
            elif j == "normalize_size_format":
                df[i] = df[i].astype(str).str.strip().str.replace(" ", "_")
                df.loc[df[i] == "", i] = np.nan
            a.append(f"{i}: {j}")

        for i, j in preprocess_plan.get("fill_missing", {}).items():
            if i not in df.columns:
              continue
            if i == target_col:
              continue

            if j == "missing_label":
                df[i] = df[i].fillna("missing_label")
            elif j == "zero":
                df[i] = df[i].fillna(0)
            elif j == "empty_string":
                df[i] = df[i].fillna("")
            elif j == "most_frequent":
                mode = df[i].mode(dropna=True)
                if not mode.empty:
                    df[i] = df[i].fillna(mode.iloc[0])
            elif j == "median":
                df[i] = df[i].fillna(df[i].median())
            elif j == "mean":
                df[i] = df[i].fillna(df[i].mean())
            elif j == "none":
                pass
            a.append(f"{i}: fill_missing -> {j}")

        for i, j in preprocess_plan.get("outlier_actions", {}).items():
            if i not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[i]):
                continue
            if j == "clip_iqr":
                q1 = df[i].quantile(0.25)
                q3 = df[i].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[i] = df[i].clip(lower=lower, upper=upper)
                a.append(f"{i}: outlier_actions -> clip_iqr")
            elif j == "winsorize_p01_p99":
                lower = df[i].quantile(0.01)
                upper = df[i].quantile(0.99)
                df[i] = df[i].clip(lower=lower, upper=upper)
                a.append(f"{i}: outlier_actions -> winsorize_p01_p99")
            elif j == "none":
                a.append(f"{i}: outlier_actions -> none")

        for i, j in preprocess_plan.get("categorical_handling", {}).items():
            if i in df.columns and j == "drop" and i != "Цена":
                df = df.drop(columns=[i])
                a.append(f"{i}: drop")
        for i, j in preprocess_plan.get("text_processing", {}).items():
            if i not in df.columns:
                continue
            if j == "drop":
                df = df.drop(columns=[i])
                a.append(f"{i}: text drop")
            elif j == "basic_clean":
              df[i] = (df[i].astype(str).str.lower().str.strip().apply(lambda x: " ".join(x.split())))
              a.append(f"{i}: basic_clean")
            elif j == "keep":
                pass

        output_path = os.path.join(ARTIFACT_DIR, "preprocessed_dataset.csv")
        df.to_csv(output_path, index=False)

        return {
            "status": "success",
            "preprocessed_dataset_path": output_path,
            "final_shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "applied_actions": a
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
