import json
import os

from Node_Memory import set_pipeline_state, get_pipeline_state


def preprocess_decision(input_data, llm):
    """
    Принимает json: eda_report (EDA-отчёт по датасету)
    Возвращает json: status, decision_source, drop_columns, fill_missing, drop_duplicates, outlier_actions, categorical_handling, text_processing, special_preprocessing, handle_imbalance, notes

    Анализирует EDA-отчёт и формирует план предобработки датасета для следующей ноды.
    Возвращает только решение по предобработке, без применения изменений к самому датасету.
    """

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

        prompt = f"""
    Ты опытный ML-инженер.
    
    Контекст задачи:
    - Бизнес-заказчик: Oskelly
    - Задача: подобрать релевантную цену для вещи по её характеристикам
    - Тип ML-задачи: regression
    - Целевая колонка: Цена (ее мы не меняем)
    
    Особенности предметной области:
    - Колонка "Размер" может содержать значения в смешанных системах, например: FR 40, IT 40, EU 40, INT M, INT XS, JEANS 31.
    - Колонка "Скидка" может содержать значения вида "-11%".
    - Не включай колонку "Цена" в словари fill_missing и outlier_actions
    - Если в колонке "Скидка" есть пропуски, считай это отсутствием скидки
    
    Тебе дан EDA-отчёт по датасету для ML-задачи.
    
    На этом этапе: очищай и стандартизируй данные, обрабатывай пропуски, дубликаты и технические колонки, выполняй специальные преобразования формата.
    Обязательно опирайся на блоки из EDA: dataset_description, missing_top, numeric_distributions, categorical_profiles, text_profiles.
    Если распределение числового признака сильно скошено (|skew| > 1.0) или есть сильные хвосты (большая разница между q99 и q75),
    рекомендуй outlier_actions для такого признака, кроме целевой колонки.
    Для высококардинальных категориальных признаков не удаляй их автоматически: оставляй keep, если признак потенциально полезен для цены.
    Для текстовых колонок по умолчанию выбирай basic_clean, а не drop, если это не явный мусор.
    
    Правила: верни только JSON. notes должны быть короткими и на русском.
    Не предлагай агрессивное удаление признаков без явного обоснования из EDA.
    
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
    """

        response = llm.invoke(prompt)
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
        return json.dumps(plan, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": str(e)
            },
            ensure_ascii=False,
            indent=2
        )
