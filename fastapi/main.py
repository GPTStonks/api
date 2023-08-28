import ast
import asyncio
import os
import uuid

import pandas as pd
import torch
from openbb_chat.classifiers.stransformer import STransformerZeroshotClassifier
from openbb_chat.llms.guidance_wrapper import GuidanceWrapper
from openbb_terminal.sdk import openbb
from rich.progress import track
from utils import get_func_parameter_names

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

results_store = {}


@app.on_event("startup")
def init_data():
    df = pd.read_csv(os.environ["CSV_PATH"], sep=os.environ["CSV_SEPARATOR"])
    df = df.dropna()
    app.descriptions = df["Descriptions"].tolist()
    app.definitions = df["Definitions"].tolist()

    app.keys = []
    for idx, descr in track(
        enumerate(app.descriptions), total=len(app.descriptions), description="Processing..."
    ):
        topics = app.definitions[idx][: app.definitions[idx].index("(")].split(".")[1:]
        if descr.find("[") != -1:
            descr = descr[: descr.find("[")].strip()
        if descr.strip()[-1] != ".":
            search_str = f"{descr.strip()}. Topics: {', '.join(topics)}."
        else:
            search_str = f"{descr.strip()} Topics: {', '.join(topics)}."
        app.keys.append(search_str)
    app.stransformer = STransformerZeroshotClassifier(app.keys, os.environ["CLASSIFIER_MODEL"])
    app.guidance_wrapper = GuidanceWrapper(model_id=os.environ["LLM_MODEL"])


@app.post("/process_query_async")
async def process_query_async(request: Request):
    data = await request.json()
    query = data.get("query")

    job_id = str(uuid.uuid4())
    response = {"message": "Processing started", "job_id": job_id}

    asyncio.create_task(run_model_in_background(job_id, query))

    return response


async def run_model_in_background(job_id: str, query: str):
    key, _, idx = app.stransformer.classify(query)

    func_def = app.definitions[idx]
    func_descr = app.descriptions[idx]

    param_names = get_func_parameter_names(func_def)
    if len(param_names) == 0:
        # No parameters, just call the function without using the LLM
        final_func_call = func_def[: func_def.index("(") + 1] + ")"
        try:
            res = eval(final_func_call)
            if isinstance(res, pd.DataFrame):
                res = res.to_dict(orient="dict")
            elif res is None:
                res = "Nothing returned."
            else:
                res = str(res)
            results_store[job_id] = {"result": res}
        except Exception as e:
            results_store[job_id] = {"error": "Error processing the query. Please try again!"}
        finally:
            return

    # Guess parameters with LLM
    param_str = ""
    param_keys = [f"param_{idx}" for idx, _ in enumerate(param_names)]
    for idx, param in enumerate(param_names):
        param_str += f"{param}" + " = {{gen " + f"'{param_keys[idx]}'" + " stop='\n'}}\n"

    template = f"""The Python function `{func_def}` is used to "{func_descr}". Given the prompt "{query}", write the correct parameters for the function using Python:
```python
{param_str[:-1]}
```"""
    program = app.guidance_wrapper(template)
    executed_program = program()

    # Call the function with inferred parameters
    inner_func_str = ",".join([executed_program[key] for key in param_keys])
    final_func_call = func_def[: func_def.index("(") + 1] + inner_func_str + ")"
    try:
        res = eval(final_func_call)
        if isinstance(res, pd.DataFrame):
            res = res.to_dict(orient="dict")
        elif res is None:
            res = "Nothing returned."
        else:
            res = str(res)
        results_store[job_id] = {"result": res}
    except Exception as e:
        results_store[job_id] = {"error": "Error processing the query. Please try again!"}
    finally:
        return


@app.get("/get_processing_result/{job_id}")
async def get_processing_result(job_id: str):
    result = results_store.get(job_id)
    if result is None:
        return {"status": "processing"}
    else:
        return {"status": "completed", "result": result}
