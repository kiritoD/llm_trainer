# Load model directly
# Use a pipeline as a high-level helper
import torch
import torch.nn.functional as F
from src.data.tokenizer import get_tokenizer
from src.peft_utils.convert import PEFT
from src.utils import get_logger, parse_arguments, rank_zero_info
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = get_logger("Chat")
path = "/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/hejunheng.123/EasyGuard/backbone/openllama_model_output_v1.3/checkpoint-13448"
tokenizer = get_tokenizer(
    {
        "pretrain_tokenizer_path": path,
        "add_special_tokens": True,
        "special_tokens": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "sep_token": "",
            "pad_token": "<pad>",
            "unk_token": "",
        },
        "max_len": 1024,
    }
)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
# peft_path = "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/applications/us_productor/outputs/llama2-13b-chat-lora/rt_modified_v3/LORA/step_100"
peft_path = r"/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/yuyang/EasyGuard_DJW/EasyGuard/examples/llm_train/model_outputs/MFashionGPT_7b_b3_v1·3_P_at_LoRA_r4/1214_1942/LORA/step_6369/"

peft = True

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

PEFT.weights_merge(model)
model.to("cuda:0")
model.eval()
instruct_default = "You are now a content control assistant, responsible for finding redirect sentences in sentences. If the content doesn't contain any guiding sentences, please answer 'no', otherwise, please extract the relevant guiding sentences in the input sentences and then use \"|\" to separate each sentence.\n"
while True:
    instruct = input("\n\nPrompt:")
    if instruct is None:
        instruct = instruct_default
    input_text = input("Please ask:")
    inputs = tokenizer(instruct + input_text + "\nanswer:")
    inputs = {_: torch.tensor([inputs[_]]).to("cuda:0") for _ in inputs}
    results = model.generate(**inputs, max_length=2048)
    print(f"\n\nAssisstant: {tokenizer.decode(results[0][1:-1])}")
