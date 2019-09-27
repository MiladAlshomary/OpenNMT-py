""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns


    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

class View(nn.Module):
    """Helper class to be used inside Sequential object to reshape Variables"""
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


class ContextLocalFeaturesProjector(nn.Module):
    """
        Reshape local image features.
    """
    def __init__(self, num_layers, nfeats, outdim, dropout,
            use_nonlinear_projection):
        """
        Args:
            num_layers (int): 1.
            nfeats (int): size of image features.
            outdim (int): size of the output dimension.
            dropout (float): dropout probablity.
            use_nonliner_projection (bool): use non-linear activation
                    when projecting the image features or not.
        """
        super(ContextLocalFeaturesProjector, self).__init__()
        assert(num_layers==1), \
                'num_layers must be equal to 1 !'
        self.num_layers = num_layers
        self.nfeats = nfeats
        self.dropout = dropout
        
        layers = []
        # # reshape input
        # layers.append( View(-1, 7*7, nfeats) )
        # linear projection from feats to rnn size
        layers.append( nn.Linear(nfeats, outdim*num_layers) )
        if use_nonlinear_projection:
            layers.append( nn.Tanh() )
        layers.append( nn.Dropout(dropout) )
        #self.batch_norm = nn.BatchNorm2d(512)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        #out = self.layers(input)
        #print( "out.size(): ", out.size() )
        #if self.num_layers>1:
        #    out = out.unsqueeze(0)
        #    out = torch.cat([out[:,:,0:out.size(2):2], out[:,:,1:out.size(2):2]], 0)
        #    #print( "out.size(): ", out.size() )
        print(input.shape)
        return input

class ContextualFeaturesProjector(nn.Module):
    """
        Project global image features using a 2-layer multi-layer perceptron.
    """
    def __init__(self, num_layers, nfeats, outdim, dropout,
            use_nonlinear_projection):
        """
        Args:
            num_layers (int): number of decoder layers.
            nfeats (int): size of image features.
            outdim (int): size of the output dimension.
            dropout (float): dropout probablity.
            use_nonliner_projection (bool): use non-linear activation
                    when projecting the image features or not.
        """
        super(ContextualFeaturesProjector, self).__init__()
        self.num_layers = num_layers
        self.nfeats = nfeats
        self.outdim = outdim
        self.dropout = dropout
        
        layers = []
        layers.append( nn.Linear(nfeats, nfeats) )
        if use_nonlinear_projection:
            layers.append( nn.Tanh() )
        layers.append( nn.Dropout(dropout) )
        # final layers projects from nfeats to decoder rnn hidden state size
        layers.append( nn.Linear(nfeats, outdim*num_layers) )
        if use_nonlinear_projection:
            layers.append( nn.Tanh() )
        layers.append( nn.Dropout(dropout) )
        #self.batch_norm = nn.BatchNorm2d(512)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        #print( "out.size(): ", out.size() )
        if self.num_layers>1:
            out = out.unsqueeze(0)
            out = torch.cat([out[:,:,0:out.size(2):2], out[:,:,1:out.size(2):2]], 0)
            #print( "out.size(): ", out.size() )
        return out

class NMTContextDModel(nn.Module):
    """
    The encoder + decoder Neural Machine Translation Model
    with image features used to initialise the decoder.
    """
    def __init__(self, encoder, decoder, context_encoder, multigpu=False):
        """
        Args:
            encoder(*Encoder): the various encoder.
            decoder(*Decoder): the various decoder.
            context_encoder(Encoder): the image encoder.
            multigpu(bool): run parellel on multi-GPU?
        """
        self.multigpu = multigpu
        super(NMTContextDModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.context_encoder = context_encoder

    def _combine_enc_state_img_proj(self, enc_hidden, img_proj):
        """
        Args:
            enc_hidden(tuple or DecoderState):
                                    Tuple containing hidden state and cell
                                    (h,c) in case of an LSTM, or a hidden state
                                    (DecoderState) in case of GRU cell.
            img_proj(Variable):     Variable containing projected image features.
            
            Returns:
                Variable with DecoderState combined with image features.
        """
        enc_init_state = []
        if isinstance(enc_hidden, tuple):
            for e in enc_hidden:
                enc_init_state.append(e + img_proj)
            enc_init_state = tuple(enc_init_state)
        else:
            enc_init_state = enc_hidden + img_proj
        return enc_init_state


    def forward(self, src, tgt, lengths, context_feats, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        
        # project image features
        feats_proj = self.context_encoder(context_feats)

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        enc_init_state = self._combine_enc_state_img_proj(enc_state, feats_proj)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_init_state)
        
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns


class NMTSrcContextModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a doubly-attentive encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, context_encoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTSrcContextModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.context_encoder = context_encoder

    def forward(self, src, tgt, lengths, context_feats, bptt=False):
        """Forward propagate a `src`, `img_feats` and `tgt` tuple for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            img_feats(FloatTensor): image features of size (len x batch x nfeats).
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """

        # project/transform local image features into the expected structure/shape
        context_feats = context_feats.unsqueeze(-1)
        context_proj = self.context_encoder( context_feats )

        tgt = tgt[:-1]  # exclude last target from inputs
        
        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        if bptt is False:
            enc_state = self.decoder.init_state(src, context, context_proj, enc_state)

        out, out_imgs, attns = self.decoder(tgt, context, context_proj, enc_state, context_lengths=lengths)
        
        if self.multigpu:
            # Not yet supported on multi-gpu
            attns = None

        return out, out_imgs, attns