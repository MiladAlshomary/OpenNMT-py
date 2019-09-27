"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder, InputFeedRNNDecoderDoublyAttentive
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder, "doubly-atten": InputFeedRNNDecoderDoublyAttentive,
           "cnn": CNNDecoder, "transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder", "InputFeedRNNDecoderDoublyAttentive"
           "InputFeedRNNDecoder", "str2dec"]
