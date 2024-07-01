from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm-6b", trust_remote_code=True
)
model = (
    AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    .half()
    .cuda()
)
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
while True:
    text = input("please input your question:")
    response, history = model.chat(tokenizer, text, history=history)
    print(response)
