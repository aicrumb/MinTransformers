# mintransformers

I wanted a little control so that I can focus on data filtering alone, no model differences besides size.

Example usage:
```python
from modeling import EncoderDecoder, Config
model = EncoderDecoder(
    Config(
        d_model = 1024,
        d_out = 256,
        num_layers = 28, # *2
        num_heads = 64,
        num_embeddings = 256,
        position_embeddings = 512,
        tie_embeddings = True
    )
)
print("init with", sum([p.numel() for p in model.parameters()]), "params")
# init with 825098752 params

model(
    torch.randint(0,255,(4,128)),
    torch.randint(0,255,(4,1))
).shape
torch.Size([4, 1, 256])
```

There is no tokenizer code provided so using one already built from HuggingFace is probably a good option.
