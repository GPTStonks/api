from typing import List, Optional
from urllib.error import HTTPError

import yfinance as yf
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from langchain_community.utilities import PythonREPL
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from openbb_chat.kernels.auto_llama_index import AutoLlamaIndex
from requests.exceptions import ReadTimeout


async def get_openbb_chat_output(
    query_str: str,
    auto_llama_index: AutoLlamaIndex,
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
) -> str:
    """Get OpenBB tool output using RAG.

    The output is obtained by:

    1. Retrieving similar nodes with hybrid search from OpenBB's documentation.
    2. Applying the defined postprocessors.
    3. Generating the response given the information in the retrieved and postprocessed nodes to the LLM.

    Args:
        query_str (`str`): input to OpenBB's tool, given by the agent's LLM.
        auto_llama_index (`AutoLlamaIndex`): contains all the necessary tools to perform the RAG.
        node_postprocessors (`Optional[List[BaseNodePostprocessor]]`): postprocessors to apply to the retrieved nodes.

    Returns:
        `str`: response by the RAG system to the given query.
    """
    nodes = await auto_llama_index.aretrieve(query_str)
    if node_postprocessors is not None:
        for node_postprocessor in node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(nodes)
    return (await auto_llama_index.asynth(str_or_query_bundle=query_str, nodes=nodes)).response


def fix_frequent_code_errors(prev_code: str, openbb_pat: Optional[str] = None) -> str:
    """Fix common errors in the LLM-generated code.

    It also adds OpenBB Personal Access Token (PAT) to authenticate the OpenBB Platform calls and use the user's data providers.

    Args:
        prev_code (`str`): code generated by the LLM.
        openbb_pat (`Optional[str]`): user's OpenBB PAT.

    Returns:
        `str`: new code with fixes and PAT included.
    """
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
    """Run REPL over the code generated by the LLM.

    Args:
        openbb_chat_output (`str`): output generated by the LLM in the agent's OpenBB Tool.
        python_repl_utility (`PythonREPL`): REPL to run the generated code with.
        openbb_pat (`Optional[str]`): user's OpenBB PAT.

    Returns:
        `str`: the output of the REPL call, which is stdout.
    """
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
