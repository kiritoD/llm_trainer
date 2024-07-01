from transformers import AutoModel, AutoTokenizer

# path = "THUDM/chatglm2-6b"
path = "/root/.cache/huggingface/hub/models--THUDM--chatglm2-6b/snapshots/b1502f4f75c71499a3d566b14463edd62620ce9f"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModel.from_pretrained(path, trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
while True:
    text = input("please input your question:")
    response, history = model.chat(tokenizer, text, history=history)
    print(response)
