from src.utils.auxiliary import model_linear_postfix
from transformers import AutoModelForCausalLM

path = "FacebookAI/roberta-base"

model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
print(model_linear_postfix(model))
