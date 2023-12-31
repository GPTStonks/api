import os
from typing import List, Optional
from urllib.error import HTTPError

import yfinance as yf
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from langchain.utilities import PythonREPL
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT_TMPL
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex
from requests.exceptions import ReadTimeout


def get_func_parameter_names(func_def: str) -> List[str]:
    # E.g. stocks(symbol: str, time: int) would be ["symbol", "time"]
    inner_func = func_def[func_def.index("(") + 1 : func_def.index(")")].strip()
    if inner_func == "":
        return []
    return [param.strip().split(":")[0].strip() for param in inner_func.split(",")]


def get_wizardcoder_select_template():
    return """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Given the request "{{query}}" and the following Python functions:
{{#each func_defs_and_descrs}}- {{this.def}}: {{this.descr}}
{{/each}}
Choose the best function and use it to solve the request.

### Response:
Here is the code you asked for:
```python
import openbb
return {{select 'func' options=func_names}}({{gen 'params' stop=')'}}
```"""


def get_wizardcoder_few_shot_template():
    return """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Given a query and a Python function, use that function with the correct parameters to return the queried information. Here are some examples:
---
Query: load the information about Apple from 2019 to 2023
Function: openbb.stocks.load(symbol: str, start_date: Union[datetime.datetime, str, NoneType] = None, interval: int = 1440, end_date: Union[datetime.datetime, str, NoneType] = None, prepost: bool = False, source: str = "YahooFinance", weekly: bool = False, monthly: bool = False, verbose: bool = True)
Code: openbb.stocks.load(symbol="AAPL", start_date="2019-01-01", end_date="2023-01-01")
---
Query: get historical prices for Amazon from 2010-01-01 to 2023-01-01
Function: openbb.stocks.ca.hist(similar: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None, candle_type: str = "a")
Code: openbb.stocks.ca.hist(similar=["AMZN"], start_date="2010-01-01", end_date="2023-01-01")
---
Query: {{query}}
Function: {{func_def}}

### Response:
Code: {{func_name}}({{gen 'params' stop=')'}}"""


def get_openllama_template():
    return """The Python function `{{func_def}}` is used to "{{func_descr}}". Given the prompt "{{query}}", write the correct parameters for the function using Python:
```python
{{param_str}}
```"""


def get_griffin_few_shot_template():
    # Template follows Griffin GPTQ format: https://huggingface.co/TheBloke/Griffin-3B-GPTQ
    return """You are a financial personal assistant called GPTStonks and you are having a conversation with a human. You are not allowed to give advice or opinions. You must decline to answer questions not related to finance. Please respond briefly, objectively and politely.
### HUMAN:
Given a query and a Python function, use that function with the correct parameters to return the queried information. Here are some examples:
1. Query: load the information about Apple from 2019 to 2023
Function: openbb.stocks.load(symbol: str, start_date: Union[datetime.datetime, str, NoneType] = None, interval: int = 1440, end_date: Union[datetime.datetime, str, NoneType] = None, prepost: bool = False, source: str = "YahooFinance", weekly: bool = False, monthly: bool = False, verbose: bool = True)
Answer: Sure! Here is the information about Apple (AAPL) from 2019-01-01 to 2023-01-01.
Code: openbb.stocks.load(symbol="AAPL", start_date="2019-01-01", end_date="2023-01-01")
2. Query: get historical prices for Amazon from 2010-01-01 to 2023-01-01
Function: openbb.stocks.ca.hist(similar: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None, candle_type: str = "a")
Answer: These are the historical prices for Amazon (AMZN) for the requested dates.
Code: openbb.stocks.ca.hist(similar=["AMZN"], start_date="2010-01-01", end_date="2023-01-01")
3. Query: {{query}}
Function: {{func_def}}

### RESPONSE:
Answer: {{gen 'answer' stop='\n' temperature=0.9 top_p=0.9}}
Code: {{func_name}}({{gen 'params' stop=')'}}
"""


def get_griffin_general_template():
    return """You are a financial personal assistant called GPTStonks and you are having a conversation with a human. You are not allowed to give advice or opinions. You must decline to answer questions not related to finance. Please respond briefly, objectively and politely.
### HUMAN:
{{query}}

### RESPONSE:
{{gen 'answer' stop='.\n' temperature=0.9 top_p=0.9}}"""


def get_definitions_path():
    return "/api/gptstonks_api/data/openbb-docs-v3.2.2-funcs.csv"


def get_definitions_sep():
    return "@"


def get_embeddings_path():
    return "/api/gptstonks_api/data/openbb-docs-v3.2.2.pt"


def get_default_classifier_model():
    return "sentence-transformers/all-MiniLM-L6-v2"


def get_default_llm():
    return "daedalus314/Griffin-3B-GPTQ"


def get_keys_file():
    return "./apikeys_list.json"


async def get_openbb_chat_output(
    query_str: str,
    auto_llama_index: AutoLlamaIndex,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
) -> str:
    nodes = await auto_llama_index.aretrieve(query_str)
    if node_postprocessors is not None:
        for node_postprocessor in node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(nodes)
    return (await auto_llama_index.asynth(str_or_query_bundle=query_str, nodes=nodes)).response


def fix_frequent_code_errors(prev_code: str, openbb_pat: Optional[str] = None) -> str:
    if "import pandas as pd" not in prev_code:
        prev_code = f"import pandas as pd\n{prev_code}"
    if "obb." in prev_code and "from openbb import obb" not in prev_code:
        prev_code = f"from openbb import obb\n{prev_code}"
    # login to openbb hub from inside the REPL if a PAT is provided
    if openbb_pat is not None:
        prev_code = prev_code.replace(
            "from openbb import obb\n",
            f"from openbb import obb\nobb.account.login(pat='{openbb_pat}')\n",
            1,
        )
    # convert generic openbb output to JSON
    prev_code = f'{prev_code}\nprint(pd.DataFrame.from_records([dict(r) for r in res.results]).to_json(orient="records"))'
    return prev_code


def run_repl_over_openbb(
    openbb_chat_output: str, python_repl_utility: PythonREPL, openbb_pat: Optional[str] = None
) -> str:
    if "```python" not in openbb_chat_output:
        # no code available to execute
        return openbb_chat_output
    code_str = (
        openbb_chat_output.split("```python")[1].split("```")[0]
        if "```python" in openbb_chat_output
        else openbb_chat_output
    )
    fixed_code_str = fix_frequent_code_errors(code_str, openbb_pat)
    # run Python and get output
    repl_output = python_repl_utility.run(fixed_code_str)
    # get OpenBB's functions called for explicability
    openbb_funcs_called = set()
    for code_line in code_str.split("\n"):
        if "obb." in code_line:
            openbb_funcs_called.add(code_line.split("obb.")[1].strip())
    openbb_platform_ref_uri = "https://docs.openbb.co/platform/reference/"
    openbb_funcs_called_str = "".join(
        [
            f"- {x} [[documentation]({openbb_platform_ref_uri}{x.split('(')[0].replace('.', '/')})]\n"
            for x in openbb_funcs_called
        ]
    )

    return (
        "> Context retrieved using OpenBB. "
        f"OpenBB's functions called:\n{openbb_funcs_called_str.strip()}\n\n"
        f"```json\n{repl_output.strip()}\n```"
    )


async def get_openbb_chat_output_executed(
    query_str: str,
    auto_llama_index: AutoLlamaIndex,
    python_repl_utility: PythonREPL,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    openbb_pat: Optional[str] = None,
) -> str:
    output_res = await get_openbb_chat_output(query_str, auto_llama_index, node_postprocessors)
    return run_repl_over_openbb(output_res, python_repl_utility, openbb_pat)


def run_qa_over_tool_output(tool_input: str | dict, llm: BaseLLM, tool: BaseTool) -> str:
    tool_output: str = tool.run(tool_input)
    model_prompt: str = PromptTemplate(
        input_variables=["context_str", "query_str"],
        template=os.getenv("CUSTOM_GPTSTONKS_QA", DEFAULT_TEXT_QA_PROMPT_TMPL),
    ).format(query_str=tool_input, context_str=tool_output)
    answer: str = llm(model_prompt)

    return f"> Context retrieved using {tool.name}.\n\n" f"{answer}"


async def arun_qa_over_tool_output(
    tool_input: str | dict, llm: BaseLLM, tool: BaseTool, original_query: Optional[str] = None
) -> str:
    tool_output: str = await tool.arun(tool_input)
    model_prompt = PromptTemplate(
        input_variables=["context_str", "query_str"],
        template=os.getenv("CUSTOM_GPTSTONKS_QA", DEFAULT_TEXT_QA_PROMPT_TMPL),
    )
    if original_query is not None:
        answer: str = await llm.apredict(
            model_prompt.format(query_str=original_query, context_str=tool_output)
        )
    else:
        answer: str = await llm.apredict(
            model_prompt.format(query_str=tool_input, context_str=tool_output)
        )

    return f"> Context retrieved using {tool.name}.\n\n" f"{answer}"


def yfinance_info_titles(tool_input: str | dict) -> str:
    company = yf.Ticker(tool_input)
    try:
        links = [n["link"] for n in company.news if n["type"] == "STORY"]
    except (HTTPError, ReadTimeout, ConnectionError):
        if not links:
            return f"No news found for company that searched with {tool_input} ticker."
    return "\n".join([f'- {new["title"]} [link]({new["link"]})' for new in company.news])
