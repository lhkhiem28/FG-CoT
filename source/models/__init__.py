from source.models.llm import *

load_model = {
    'llm': BaselineLLM,
}

# Replace the following with the model paths
# Qwen3 / Qwen3.5 are hybrid thinking models; llm.py prefills the empty <think></think>
# block for them, which is what enable_thinking=False does in their chat template
get_llm_path = {
    'qwen2.5-14b'     : 'Qwen/Qwen2.5-14B-Instruct'                 ,
    'qwen2.5-32b'     : 'Qwen/Qwen2.5-32B-Instruct'                 ,
    'qwen3-14b'       : 'Qwen/Qwen3-14B'                            ,
    'qwen3-32b'       : 'Qwen/Qwen3-32B'                            ,
    'llama-3.1-8b'    : 'meta-llama/Llama-3.1-8B-Instruct'          ,
    'llama-3.1-70b'   : 'meta-llama/Llama-3.1-70B-Instruct'         ,
}