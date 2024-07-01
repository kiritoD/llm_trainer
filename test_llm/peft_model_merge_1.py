# Load model directly
# Use a pipeline as a high-level helper
import torch
import torch.nn.functional as F
from src.data.tokenizer import get_tokenizer
from src.peft_utils.convert import PEFT
from src.utils import get_logger, parse_arguments, rank_zero_info
from transformers import AutoModelForCausalLM, AutoTokenizer


# 保持一致， transformers.trainer_utils.set_seed
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_torch_npu_available():
        torch.npu.manual_seed_all(seed)
    if is_tf_available():
        tf.random.set_seed(seed)


# set_seed(42)


logger = get_logger("Chat")
# path = "/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/hejunheng.123/EasyGuard/backbone/Mistral-7B-s2-T30B-CT-v1.1_new/checkpoint-5101"
path = "/mnt/bn/ecom-v6arnold2-yuyang/wangpeiwen/wpw/Models/openllama/openllama_base/checkpoint-13448"
# tokenizer = get_tokenizer(
#     {
#         "pretrain_tokenizer_path": path,
#         "add_special_tokens": True,
#         "special_tokens": {
#             "bos_token": "<s>",
#             "eos_token": "</s>",
#             "sep_token": "",
#             "pad_token": "<pad>",
#             "unk_token": "<unk>",
#             "additional_special_tokens": [
#                 "<approve>",
#                 "<unknow>",
#                 "<BI>",
#                 "<MFE>",
#                 "<IP>",
#                 "<AT>",
#                 "<PCP>",
#                 "<RT>",
#                 "<FP>",
#                 "<LCC>",
#                 "<BASB>",
#                 "<ND>",
#                 "<UCB>",
#                 "<VB>",
#                 "<VB>",
#                 "<DAB>",
#                 "<PCLR>",
#                 "<DE>",
#                 "<GRB>",
#                 "<BI>",
#                 "<MP>",
#                 "<IAC>",
#                 "<HRPP>",
#                 "<IIOL>",
#                 "<SP>",
#                 "<IA>",
#                 "<MS>",
#                 "<BASB>",
#                 "<ML>",
#                 "<BIC>",
#                 "<MP>",
#                 "<MGW>",
#                 "<MO>",
#                 "<IB>",
#                 "<BI>",
#                 "<PC>",
#                 "<MRRS>",
#                 "<MC>",
#                 "<ALDT>",
#                 "<AC>",
#             ],
#         },
#         "max_len": 2048,
#     }
# )
tokenizer = get_tokenizer(
    {
        "pretrain_tokenizer_path": path,
        "add_special_tokens": True,
        "special_tokens": {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "sep_token": "",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        },
        "max_len": 2048,
    }
)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
# peft_path = "/hpc2hdd/home/fwang380/dongjunwei/llm_trainer/applications/us_productor/outputs/llama2-13b-chat-lora/rt_modified_v3/LORA/step_100"
# peft_path = r"/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/huotianjiao/output/checkpoint-1000/"
# peft_path = '/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/hewenyao/ft_models/VOC/llama2-13b-chat/benchmark/experiment0110_lora_mistral_0120_2024_01_18_11_06_47/checkpoint-873'
peft_path = '/mnt/bn/ecom-v6arnold2-yuyang/wangpeiwen/wpw/EasyGuard/backbone/product/2_sft/sft_ki/LORA/step_19'
# peft_path = r"/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/qinguanqiu/test/EasyGuard/examples/llm_train//applications/audit/outputs/7b_zh_s6_231201_1220_max2048/LORA/step_2734/"

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
model.half()
model.to("cuda:0")
model.eval()

# model.base_model.model.save_pretrained(
#     "/mnt/bn/ecom-ccr-dev-878d5f0f/mlx/users/qinguanqiu/test/EasyGuard/examples/llm_train//applications/audit/outputs/7b_zh_s6_231201_1220_max2048/",
#     from_pt=True,
# )
model.base_model.model.save_pretrained("/mnt/bn/ecom-v6arnold2-yuyang/wangpeiwen/wpw/EasyGuard/backbone/product/2_sft/sft_ki/LORA/merge_lora", from_pt=True)
