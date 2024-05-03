import time
import torch
import transformers
from chat_cli_cllm import jacobi_generate

model_path = 'cllm/consistency-llm-7b-sharegpt48k'

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_path, padding_side="right"
)

config = transformers.AutoConfig.from_pretrained(model_path)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    load_in_8bit=True
)

system_prompt = "Answer in English unless other language is used. A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
roles = ("USER", "ASSISTANT") #support vicuna
user_input = "Which methods did Socrates employ to challenge the prevailing thoughts of his time?"
prompt = system_prompt + f"{roles[0]}: " + f"{user_input}\n{roles[1]}: "

inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')

start_time = time.time()

greedy_output, avg_fast_forward_count = jacobi_generate(
    inputs, model, tokenizer,
    max_new_tokens=16, max_seq_len=1024,
    dev='cuda:0'
)
output_text = tokenizer.decode(greedy_output[0])
print(output_text)

time_delta = time.time() - start_time
cnt_tokens = len(greedy_output[0])
print('e2e speed:', time_delta, cnt_tokens, cnt_tokens / time_delta)
