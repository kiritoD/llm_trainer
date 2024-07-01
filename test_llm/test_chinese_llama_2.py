# Load model directly
# Use a pipeline as a high-level helper
import torch
import torch.nn.functional as F
from src.data.tokenizer import get_tokenizer
from src.peft_utils.convert import PEFT
from src.utils import get_logger, parse_arguments, rank_zero_info
from src.utils.auxiliary import model_linear_postfix
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE_WITH_SYSTEM_PROMPT = "[INST] <<SYS>>\n" "{system_prompt}\n" "<</SYS>>\n\n" "{instruction} [/INST]"

path = "hfl/chinese-alpaca-2-7b"
logger = get_logger("Chat")

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
print(model_linear_postfix(model))
quit()
peft_path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/applications/us_productor/outputs/llama2-13b-chat-lora/rt_modified_v3/LORA/step_100"

peft = False

#  model preprocess
try:
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
except:
    ...
if peft:
    logger.info("start to use peft to process model")
    model = PEFT.from_pretrained(
        model,
        peft_path,
    )

model.to("cuda:0")
model.eval()
instruct_default = "你是一个医生，请回答我的提问.\n"
while True:
    instruct = input("\n\nPrompt:")
    if instruct is None:
        instruct = instruct_default
    input_text = input("Please ask:")
    inputs = tokenizer(instruct + input_text + "\nanswer:")
    inputs = {_: torch.tensor([inputs[_]]).to("cuda:0") for _ in inputs}
    results = model.generate(**inputs, max_length=2048)
    print(f"\n\nAssisstant: {tokenizer.decode(results[0][1:-1])}")
