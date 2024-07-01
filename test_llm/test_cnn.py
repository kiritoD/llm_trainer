from datasets import load_dataset
from evaluate import evaluator

task_evaluator = evaluator("summarization")
data = load_dataset("cnn_dailymail", "3.0.0", split="validation[:40]")
results = task_evaluator.compute(
    model_or_pipeline="facebook/bart-large-cnn",
    data=data,
    input_column="article",
    label_column="highlights",
)
