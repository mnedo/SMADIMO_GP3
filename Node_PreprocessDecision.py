import json


def preprocess_decision(input_data, llm):
    """
    Принимает json: eda_report (EDA-отчёт по датасету)
    Возвращает json: status, decision_source, drop_columns, fill_missing, drop_duplicates, outlier_actions, categorical_handling, text_processing, special_preprocessing, handle_imbalance, notes

    Анализирует EDA-отчёт и формирует план предобработки датасета для следующей ноды.
    Возвращает только решение по предобработке, без применения изменений к самому датасету.
    """

    try:
        data = json.loads(input_data)
        eda_report = data["eda_report"]

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
    
    На этом этапе: очищай и стандартизируй данные, обрабатывай пропуски, дубликаты и технические колонки, выполняй специальные преобразования формата
    
    Правила: верни только JSON.notes должны быть короткими и на русском
    
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
