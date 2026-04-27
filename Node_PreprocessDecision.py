import json
import os

from Node_Memory import set_pipeline_state, get_pipeline_state, record_llm_call


def preprocess_decision(input_data, llm):
    """
    Принимает json: eda_report (EDA-отчёт по датасету)
    Возвращает json: status, decision_source, drop_columns, fill_missing, drop_duplicates, outlier_actions, categorical_handling, text_processing, special_preprocessing, handle_imbalance, notes

    Анализирует EDA-отчёт и формирует план предобработки датасета для следующей ноды.
    Возвращает только решение по предобработке, без применения изменений к самому датасету.
    """

    response = None
    try:
        if isinstance(input_data, str):
            data = json.loads(input_data)
        elif isinstance(input_data, dict):
            data = input_data
        else:
            return json.dumps(
                {"status": "error", "message": "Неподдерживаемый тип входа для preprocess_decision"},
                ensure_ascii=False,
                indent=2
            )

        eda_report = data.get("eda_report")
        if eda_report is None:
            eda_report_path = data.get("eda_report_path") or get_pipeline_state().get("eda_report_path")
            if not eda_report_path or not os.path.exists(eda_report_path):
                return json.dumps(
                    {"status": "error", "message": "Не передан eda_report и не найден eda_report_path в state"},
                    ensure_ascii=False, indent=2
                )
            with open(eda_report_path, encoding="utf-8") as f:
                eda_report = json.load(f)

        with open("prompts.json", encoding="utf-8") as f:
            prompts_data = json.load(f)
        USER_PROMPT = prompts_data[prompts_data["PROMPT_STYLE"]]["preprocess_decision"]
        USER_PROMPT += f"""

Правила: верни только JSON. notes должны быть короткими и на русском.
Не предлагай агрессивное удаление признаков без явного обоснования из EDA. Не включай таргет колонку "Цена" в словари fill_missing и outlier_actions (для этих словарей работаем только с признаками)
    

Разрешённые значения:
- drop_columns.reason: technical_id, high_missingness, near_constant, leakage, irrelevant_for_prediction
- fill_missing: most_frequent, missing_label, median, mean, zero, empty_string, none
- outlier_actions: none, clip_iqr, winsorize_p01_p99
- categorical_handling: keep, drop
- text_processing: keep, basic_clean, drop
- special_preprocessing: percent_to_float, normalize_size_format, none

Формат ответа:
{{
  "status": "success",
  "decision_source": "llm",
  "drop_columns": [
    {{
      "column": "string",
      "reason": "technical_id | high_missingness | near_constant | leakage | irrelevant_for_prediction"
    }}
  ],
  "fill_missing": {{
    "column_name": "most_frequent | missing_label | median | mean | zero | empty_string | none"
  }},
  "drop_duplicates": true,
  "outlier_actions": {{
    "column_name": "none | clip_iqr | winsorize_p01_p99"
  }},
  "categorical_handling": {{
    "column_name": "keep | drop"
  }},
  "text_processing": {{
    "column_name": "keep | basic_clean | drop"
  }},
  "special_preprocessing": {{
    "column_name": "percent_to_float | normalize_size_format | none"
  }},
  "handle_imbalance": null,
  "notes": ["string"]
}}

EDA-отчёт:
{json.dumps(eda_report, ensure_ascii=False, indent=2)}
""".strip()

        response = llm.invoke([{"role": "user", "content": USER_PROMPT}])
        content = response.content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        plan = json.loads(content)

        ARTIFACT_DIR = "artifacts"
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        preprocess_plan_path = os.path.join(ARTIFACT_DIR, "preprocess_plan.json")

        with open(preprocess_plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii = False, indent=2)

        set_pipeline_state(preprocess_plan = plan, preprocess_plan_path = preprocess_plan_path)
        record_llm_call(response, success=True)
        return json.dumps(plan, ensure_ascii=False, indent=2)

    except Exception as e:
        if response is not None:
            record_llm_call(response, success=False)
        return json.dumps(
            {
                "status": "error",
                "message": str(e)
            },
            ensure_ascii=False,
            indent=2
        )
