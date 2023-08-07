from torch.nn import TransformerEncoderLayer, TransformerEncoder, Embedding, Linear
from torch.nn import TransformerDecoderLayer, TransformerDecoder
from torch import nn
import torch

"""

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

"""

class Config:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.t = Embedding(config.num_embeddings, config.d_model)
        self.p = Embedding(config.position_embeddings, config.d_model)
        self.w = config.model_class(
              config.layer_class(config.d_model,config.num_heads,config.d_model*4),
              config.num_layers
        )
        self.o = Linear(config.d_model, config.d_out)
    def forward(self, ids, hidden_states=None):
        b, t = ids.shape
        x = self.t(ids) + self.p(torch.arange(0, t))
        if isinstance(self.w, TransformerDecoder):
            output = self.o(self.w(x,x if hidden_states is not None else hidden_states))
            return output
        output = self.o(self.w(x))
        return output

class Encoder(nn.Module):
    """
        bert-like
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.layer_class = TransformerEncoderLayer
        self.config.model_class = TransformerEncoder
        self.model = Transformer(self.config)
    def forward(self, ids):
        return self.model(ids)

class Decoder(nn.Module):
    """
        gpt-like
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.layer_class = TransformerDecoderLayer
        self.config.model_class = TransformerDecoder
        self.model = Transformer(self.config)
    def forward(self, ids, hidden_states=None):
        return self.model(ids, hidden_states)

class EncoderDecoder(nn.Module):
    """
        t5-like
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        if config.tie_embeddings:
            self.decoder.model.t = self.encoder.model.t
            self.decoder.model.p = self.encoder.model.p
    def forward(self, ids_encoder, ids_decoder):
        encoder_output = self.encoder(ids_encoder)
        decoder_output = self.decoder(ids_decoder, encoder_output)
        return decoder_output
