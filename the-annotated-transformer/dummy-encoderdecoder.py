import torch
import torch.nn as nn

class DummyEncoder(nn.Module):
    def forward(self, src, src_mask):
        print("Encoding:", src)
        return src * 2

class DummyDecoder(nn.Module):
    def forward(self, memory, tgt, src_mask, tgt_mask):
        print("Decoding:", memory, tgt)
        return memory + tgt

class DummyGenerator(nn.Module):
    def forward(self, x):
        return x.sum(dim=-1)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encoder(src, src_mask)
        decoded = self.decoder(memory, tgt, src_mask, tgt_mask)
        return self.generator(decoded)

model = EncoderDecoder(DummyEncoder(), DummyDecoder(), DummyGenerator())
src = torch.tensor([[1.0, 2.0]])
tgt = torch.tensor([[0.5, 0.3]])
out = model(src, tgt, None, None)
print("Result: ", out)
