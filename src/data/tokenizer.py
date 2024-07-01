from transformers import AutoTokenizer, LlamaTokenizer


def get_tokenizer(params):
    filename = params["pretrain_tokenizer_path"]
    
    trust_remote_code = params.get("trust_remote_code", True)
    fast = params.get("use_fast", True)
    cache_dir = params.get("cache_dir", None)
    revision = params.get("revision", "main")
    token = params.get("token", None)
    
    
    tokenizer = AutoTokenizer.from_pretrained(
        filename, 
        fast=fast,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code
    )

    if params.get("add_special_tokens", False):
        special_tokens = params["special_tokens"]
        for token in special_tokens:
            if special_tokens[token] not in ["", None]:
                tokenizer.add_special_tokens({token: special_tokens[token]})

    return tokenizer
