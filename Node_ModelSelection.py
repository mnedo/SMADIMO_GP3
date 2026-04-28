import json



from Node_Memory import set_pipeline_state, get_pipeline_state, record_llm_call



_SYSTEM_MODEL_SELECTION = (

    "Ты эксперт в области машинного обучения. Отвечай только валидным JSON без пояснений и без ```json```. "

    'Формат ответа: [{"model": "НазваниеМодели", "reason": "обоснование"}]. '

    "Выбирай только из этих моделей: Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor."

)





def model_selection(input_str: str, llm=None) -> dict:

    '''

    Принимает json: dataset_path (путь к файлу), context (контекст по датасету от предыдущих нод)

    Возвращает json: status, output_path (путь к тому же файлу для следующей ноды), recommended_models (список рекомендуемых моделей с обоснованиями), context (обновленный контекст по датасету для следующих нод)



    На основе характеристик задачи предлагает список релевантных моделей для обучения с обоснованием

    '''

    try:

        if isinstance(input_str, str):

            params = json.loads(input_str)

        elif isinstance(input_str, dict):

            params = input_str

        else:

            return {

                "status": "error",

                "error": "Неподдерживаемый тип входа для model_selection"

            }

        state = get_pipeline_state()

        path = params.get('dataset_path') or state.get('featured_dataset_path') or state.get('preprocessed_dataset_path') or state.get('dataset_path')

        if not path:

            return {"status": "error", "error": "dataset_path не найден ни в аргументах, ни в pipeline_state"}



        context = params.get('context', '')



        if llm is None:

            return {

                "status": "error",

                "error": "LLM не передан в model_selection"

            }



        with open("prompts.json", encoding="utf-8") as f:

            prompts_data = json.load(f)

        USER_PROMPT = prompts_data[prompts_data["PROMPT_STYLE"]]["model_selection"]

        USER_PROMPT += f"""



Доступные модели: Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor.

Контекст: {context}



Ответ — только валидный JSON-список вида [{{"model": "...", "reason": "..."}}], без ```json и без пояснений.

""".strip()



        DEFAULT_MODELS = [

            {"model": "Ridge", "reason": "baseline"},

            {"model": "RandomForestRegressor", "reason": "nonlinear"},

            {"model": "GradientBoostingRegressor", "reason": "strong tabular"},

        ]

        models = None

        for _ in range(3):

            response = None

            try:

                response = llm.invoke([

                    {"role": "system", "content": _SYSTEM_MODEL_SELECTION},

                    {"role": "user", "content": USER_PROMPT},

                ])

                content = response.content.strip().replace("```json", "").replace("```", "").strip()

                parsed = json.loads(content)

                if isinstance(parsed, list) and len(parsed) > 0:

                    models = parsed

                    record_llm_call(response, success=True)

                    break

                record_llm_call(response, success=False)

            except Exception:

                if response is not None:

                    record_llm_call(response, success=False)

                continue

        if models is None:

            models = DEFAULT_MODELS



        set_pipeline_state(recommended_models=models)

        print('model_selection завершилась успешно')

        return {

            'status': 'ok',

            'output_path': path,

            'recommended_models': models,

            'context': context + f'\n\nModelSelection определил список релевантных моделей для обучения: {models}'

        }



    except Exception as e:

        return {

            'status': 'error',

            'error': str(e)

        }


