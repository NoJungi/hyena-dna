import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

import src.models.nn.utils as U
import src.utils as utils
import src.utils.config
import src.utils.train

log = src.utils.train.get_logger(__name__)


class Decoder(nn.Module):
    """This class doesn't do much but just signals the interface that Decoders are expected to adhere to
    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset

        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)


class SequenceDecoder(Decoder):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last"
    ):
        super().__init__()

        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None, mask=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool":
            if mask is None:
                restrict = lambda x: (
                    torch.cumsum(x, dim=-2)
                    / torch.arange(
                        1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                    ).unsqueeze(-1)
                )[..., -l_output:, :]           
            else:
                # sum masks
                mask_sums = torch.sum(mask, dim=-1).squeeze() - 1  # for 0 indexing

                # convert mask_sums to dtype int
                mask_sums = mask_sums.type(torch.int64)

                restrict = lambda x: (
                    torch.cumsum(x, dim=-2)
                    / torch.arange(
                        1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                    ).unsqueeze(-1)
                )[torch.arange(x.size(0)), mask_sums, :].unsqueeze(1)  # need to keep original shape

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        return self.output_transform(x)

class LinearSigmoidDecoder(nn.Module):
    """1 simple linear layer that transform the ebeddings into class predictions for
    each position of the sequence.
    """

    def __init__(
        self, d_model, d_output=9):

        super().__init__()
        self.d_model = d_model
        self.d_output = d_output

        self.activation = nn.Sigmoid

        self.dense_layer_activation = nn.Sequential(
            nn.Linear(self.d_model, self.d_output),
            self.activation,
        )

    def forward(self, x):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """
        x = self.dense_layer_activation(x)
        return x

    def step(self, x, state=None):
        # Ignore all length logic
        return self.forward(x)

    
class TransposeLayer(nn.Module):
    """
    From BEND. Needed for CNN_BEND decoder   
    A layer that transposes the input.
    """
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        """
        Transpose the input.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transposed tensor.
        """
        x = torch.transpose(x, 1, 2)
        return x

class CNN_BEND_Decoder(nn.Module):
    """
    Adapted from BEND. 
    A two-layer CNN with step size 1, GeLU activation, and a linear layer.
    """
    def __init__(self, d_model, 
                 d_output=9, 
                 hidden_size=64, 
                 kernel_size=3):
        """
        Build a two-layer CNN with step size 1, GeLU activation, and a linear layer.

        Parameters
        ----------
        d_model: int
            The embedding size of the input sequence.
        d_output: int
            The size of the output sequence.
        hidden_size: int
            The number of channels in the convolutional layers.
        kernel_size: int
            The kernel size of the convolutional layers.
        """
        super().__init__()
        self.d_output = d_output
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.conv1 = nn.Sequential(TransposeLayer(), 
                                   nn.Conv1d(self.d_model, self.hidden_size, self.kernel_size, stride = 1, padding = 1), 
                                   TransposeLayer(),
                                   nn.GELU())
        
        self.conv2 = nn.Sequential(TransposeLayer(), 
                                   nn.Conv1d(self.hidden_size, self.hidden_size, self.kernel_size, stride = 1, padding = 1), 
                                   TransposeLayer(), 
                                   nn.GELU(),
                                  )

        self.linear = nn.Sequential(nn.Linear(self.hidden_size, self.d_output))
        # no activation at the end like in BEND 
        
    def forward(self, x):
        """
        Forward pass of the CNN.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor. Should have shape (batch_size, length, d_model).
        Returns
        -------
        torch.Tensor
            Output tensor. Has shape (batch_size, length, d_output).

        """
        # 1st conv layer
        x = self.conv1(x)
        # 2nd conv layer 
        x = self.conv2(x)
        # linear layer 
        x = self.linear(x)
        return x
    
    def step(self, x, state=None):
        # Ignore all length logic
        return self.forward(x)
    

class TokenDecoder(Decoder):
    """Decoder for token level classification"""
    def __init__(
        self, d_model, d_output=3
    ):
        super().__init__()

        self.output_transform = nn.Linear(d_model, d_output)

    def forward(self, x, state=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """
        x = self.output_transform(x)
        return x


class NDDecoder(Decoder):
    """Decoder for single target (e.g. classification or regression)"""
    def __init__(
        self, d_model, d_output=None, mode="pool"
    ):
        super().__init__()

        assert mode in ["pool", "full"]
        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        self.mode = mode

    def forward(self, x, state=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.mode == 'pool':
            x = reduce(x, 'b ... h -> b h', 'mean')
        x = self.output_transform(x)
        return x

class StateDecoder(Decoder):
    """Use the output state to decode (useful for stateful models such as RNNs or perhaps Transformer-XL if it gets implemented"""

    def __init__(self, d_model, state_to_tensor, d_output):
        super().__init__()
        self.output_transform = nn.Linear(d_model, d_output)
        self.state_transform = state_to_tensor

    def forward(self, x, state=None):
        return self.output_transform(self.state_transform(state))


class RetrievalHead(nn.Module):
    def __init__(self, d_input, d_model, n_classes, nli=True, activation="relu"):
        super().__init__()
        self.nli = nli

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise NotImplementedError

        if (
            self.nli
        ):  # Architecture from https://github.com/mlpen/Nystromformer/blob/6539b895fa5f798ea0509d19f336d4be787b5708/reorganized_code/LRA/model_wrapper.py#L74
            self.classifier = nn.Sequential(
                nn.Linear(4 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, n_classes),
            )
        else:  # Head from https://github.com/google-research/long-range-arena/blob/ad0ff01a5b3492ade621553a1caae383b347e0c1/lra_benchmarks/models/layers/common_layers.py#L232
            self.classifier = nn.Sequential(
                nn.Linear(2 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, d_model // 2),
                activation_fn,
                nn.Linear(d_model // 2, n_classes),
            )

    def forward(self, x):
        """
        x: (2*batch, dim)
        """
        outs = rearrange(x, "(z b) d -> z b d", z=2)
        outs0, outs1 = outs[0], outs[1]  # (n_batch, d_input)
        if self.nli:
            features = torch.cat(
                [outs0, outs1, outs0 - outs1, outs0 * outs1], dim=-1
            )  # (batch, dim)
        else:
            features = torch.cat([outs0, outs1], dim=-1)  # (batch, dim)
        logits = self.classifier(features)
        return logits


class RetrievalDecoder(Decoder):
    """Combines the standard FeatureDecoder to extract a feature before passing through the RetrievalHead"""

    def __init__(
        self,
        d_input,
        n_classes,
        d_model=None,
        nli=True,
        activation="relu",
        *args,
        **kwargs
    ):
        super().__init__()
        if d_model is None:
            d_model = d_input
        self.feature = SequenceDecoder(
            d_input, d_output=None, l_output=0, *args, **kwargs
        )
        self.retrieval = RetrievalHead(
            d_input, d_model, n_classes, nli=nli, activation=activation
        )

    def forward(self, x, state=None, **kwargs):
        x = self.feature(x, state=state, **kwargs)
        x = self.retrieval(x)
        return x

class PackedDecoder(Decoder):
    def forward(self, x, state=None):
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Decoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "sequence": SequenceDecoder,
    "nd": NDDecoder,
    "retrieval": RetrievalDecoder,
    "state": StateDecoder,
    "pack": PackedDecoder,
    "token": TokenDecoder,
    "linear_sigmoid": LinearSigmoidDecoder,
    "CNN_BEND": CNN_BEND_Decoder,
}
model_attrs = {
    "linear": ["d_model"],
    "sequence": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
    "token": ["d_output"],
    "linear_sigmoid": ["d_model"],
    "CNN_BEND": ["d_model"],
}

dataset_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output", "l_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_output"],
    "forecast": ["d_output", "l_output"],
    "token": ["d_output"],
    "linear_sigmoid": ["d_output"],
    "CNN_BEND": ["d_output"],
}


def _instantiate(decoder, model=None, dataset=None):
    """Instantiate a single decoder"""
    if decoder is None:
        return None

    if isinstance(decoder, str):
        name = decoder
    else:
        name = decoder["_name_"]

    # Extract arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    # Instantiate decoder
    obj = utils.instantiate(registry, decoder, *model_args, *dataset_args)
    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
