""" ContextGate module """
import torch
import torch.nn as nn

def context_gate_factory(gate_type, embeddings_size, decoder_size,
                         attention_size, output_size, user_context_size=None):
    """Returns the correct ContextGate class"""

    gate_types = {'source': SourceContextGate,
                  'target': TargetContextGate,
                  'both': BothContextGate}

    assert gate_type in gate_types, "Not valid ContextGate type: {0}".format(
        gate_type)
    return gate_types[gate_type](embeddings_size, decoder_size, attention_size, user_context_size,
                                 output_size)

class ContextGate(nn.Module):
    """
    Context gate is a decoder module that takes as input the previous word
    embedding, the current decoder state and the attention state, and
    produces a gate.
    The gate can be used to select the input from the target side context
    (decoder state), from the source context (attention state) or both.
    """

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size, user_context_size=None):
        super(ContextGate, self).__init__()
        if user_context_size != None:
            input_size = embeddings_size + decoder_size + attention_size + user_context_size
        else:
            input_size = embeddings_size + decoder_size + attention_size

        self.gate = nn.Linear(input_size, output_size, bias=True)
        self.sig = nn.Sigmoid()

        if user_context_size != None:
            self.source_proj = nn.Linear(attention_size + user_context_size, output_size)
        else:
            self.source_proj = nn.Linear(attention_size, output_size)

        self.target_proj = nn.Linear(embeddings_size + decoder_size,
                                     output_size)

    def forward(self, prev_emb, dec_state, attn_state, user_context=None):
        if user_context is not None:
            input_tensor = torch.cat((prev_emb, dec_state, attn_state, user_context), dim=1)
            src_tensor   = torch.cat((attn_state, user_context), dim=1)
        else:
            input_tensor = torch.cat((prev_emb, dec_state, attn_state), dim=1)
            src_tensor   = attn_state

        
        tgt_tensor   = torch.cat((prev_emb, dec_state), dim=1)
        z = self.sig(self.gate(input_tensor))
        proj_source = self.source_proj(src_tensor)
        proj_target = self.target_proj(tgt_tensor)

        return z, proj_source, proj_target

class SourceContextGate(nn.Module):
    """Apply the context gate only to the source context"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size, user_context_size=None):
        super(SourceContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size,
                                        attention_size, user_context_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state, user_context=None):
        z, source, target = self.context_gate(
            prev_emb, dec_state, attn_state, user_context)
        return self.tanh(target + z * source)


class TargetContextGate(nn.Module):
    """Apply the context gate only to the target context"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size, user_context_size=None):
        super(TargetContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size,
                                        attention_size, user_context_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state, user_context=None):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state, user_context)
        return self.tanh(z * target + source)


class BothContextGate(nn.Module):
    """Apply the context gate to both contexts"""

    def __init__(self, embeddings_size, decoder_size,
                 attention_size, output_size, user_context_size=None):
        super(BothContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size,
                                        attention_size, user_context_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state, user_context=None):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state, user_context)
        return self.tanh((1. - z) * target + z * source)