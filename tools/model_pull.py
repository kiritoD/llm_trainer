# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

roberta_base = "FacebookAI/roberta-base"
# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
# model = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base")
roberta_large = "FacebookAI/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(roberta_large)
model = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-large")