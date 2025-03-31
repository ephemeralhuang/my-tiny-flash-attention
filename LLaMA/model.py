import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # number of heads for queries
    n_kv_heads: Optional[int] = None  # number of heads for keys/values
    vocab_size: int = -1  # this will be set when loading the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-6

    # Need for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # shape: (dim,)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        # (batch_size, seq_len, dim) * (batch_size, seq_len, 1) -> (batch_size, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        # (dim,) * (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        #  a_i = w_i * x_i / sqrt(mean(x_i^2) + eps)
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * model_args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if model_args.ffn_dim_multiplier is not None:
            hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)

        hidden_dim = model_args.multiple_of * (
            (hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of
        )

        # FFN_SwiGLU(x, w1, w2, w3) = SwiGLU(x, w1, w2, w3) = (w2 @ SwiGLU(w1 @ x, w3 @ x))
        self.w1 = nn.Linear(model_args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, model_args.dim, bias=False)
        self.w3 = nn.Linear(model_args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_theta_pos_frequencies(
    head_dim: int, max_seq_len: int, device: str, theta: float = 10000.0
) -> torch.Tensor:
    """
    Precompute the theta positions frequencies for the current position.

    Args:
        dim (int): The dimension of the input tensor.
        max_seq_len (int): The maximum sequence length.
        device (str): The device to run the computation on.
        theta (float, optional): The theta value for the sine and cosine functions. Default is 10000.0.

    Returns:
        torch.Tensor: The precomputed theta positions frequencies.
    """
    assert head_dim % 2 == 0, "Dimension must be even"

    # build the theta parameters
    # according to the formula: theta_i = 10000 ^ (-2(i / head_dim)) for i in [0, 1, 2, ..., head_dim / 2 - 1]
    theta_numerator = torch.arange(0, head_dim, 2).float()  # shape: (head_dim / 2,)
    theta = 1.0 / (theta ** (theta_numerator / head_dim))  # shape: (head_dim / 2,)

    # Construct the theta parameters for all positions
    m = torch.arange(max_seq_len, device=device)  # shape: (max_seq_len,)

    # Multiply each theta by each position using outer product
    # Shape: (max_seq_len,) outer_product (head_dim / 2,) -> (max_seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()

    # We can compute complex numbers in the polar form c = R * e^(i * theta), where R = 1 as follows:
    # c = cos(m * theta) + i * sin(m * theta)
    # Shape: (max_seq_len, head_dim / 2)

    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex


def apply_rotary_pos_embedding(
    x: torch.Tensor, freqs_complex: torch.Tensor, device: str
) -> torch.Tensor:
    """
    Apply the rotary positional embedding to the input tensor.
    Args:
        x (torch.Tensor): The input tensor.
        freqs_complex (torch.Tensor): The precomputed theta positions frequencies.
        device (str): The device to run the computation on.

    Returns:
        torch.Tensor: The input tensor with the rotary positional embedding applied.
    """
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex numbers
    # Two consecutive values will become a complex number
    # Shape: (batch_size, seq_len, head_dim) --reshape--> (batch_size, seq_len, head_dim / 2, 2) --> view_as_complex --> (batch_size, seq_len, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Reshape the freqs_complex to match the shape of x_complex tensor
    # Shape: (max_seq_len, head_dim / 2) --unsqueeze--> (1, max_seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Multiply each complex number in x_complex by the corresponding complex number in freqs_complex
    # Shape: (batch_size, seq_len, head_dim / 2) * (1, max_seq_len, 1, head_dim / 2) -> (batch_size, seq_len, head_dim / 2)
    x_rotated = x_complex * freqs_complex

    # Convert the complex numbers back to the real number
    # Shape: (batch_size, seq_len, head_dim / 2) --view_as_real--> (batch_size, seq_len, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)

    # Reshape the output to the original shape
    # Shape: (batch_size, seq_len, head_dim / 2, 2) --reshape--> (batch_size, seq_len, head_dim)
    x_out = x_out.reshape_as(x)

    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_heads_q (int): Number of query heads.
            n_rep (int): Number of repetitions for key and value heads.
            head_dim (int): Dimension size of each attention head.
            wq : Linear transformation for queries.
            wk : Linear transformation for keys.
            wv : Linear transformation for values.
            wo : Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.args = args

        # Indicate the number of heads for the key/value
        self.n_heads_q = args.n_heads
        # Indicate the number of heads for the query
        self.n_kv_heads = args.n_kv_heads
        # Indicate the number of repetitions for the key/value heads
        self.n_rep = args.n_heads_q // self.n_kv_heads

        # Indicate the dimension size of each attention head
        self.head_dim = args.dim // args.n_heads

        # Initialize the linear transformations for the queries, keys, and values
        self.wq = nn.Linear(args.dim, args.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads_q * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, start_pos: int):
        """
        Forward pass through the Attention module.

        Args:
            x (torch.Tensor): The input tensor.
            freqs_complex (torch.Tensor): The precomputed theta positions frequencies.
            start_pos (int): The starting position of the input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        _bsz, _seqlen, _ = x.shape  # Shape: (batch_size, 1, dim)

        # (batch_size, 1, dim) -> (batch_size, 1, n_heads_q * head_dim)
        xq = self.wq(x)
        # (batch_size, 1, dim) -> (batch_size, 1, n_kv_heads * head_dim)
        xk = self.wk(x)
        # (batch_size, 1, dim) -> (batch_size, 1, n_kv_heads * head_dim)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(_bsz, _seqlen, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(_bsz, _seqlen, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(_bsz, _seqlen, self.n_kv_heads, self.head_dim)

        # Apply the rotary positional embedding to the queries and keys
        xq = apply_rotary_pos_embedding(xq, freqs_complex, self.args.device)
        xk = apply_rotary_pos_embedding(xk, freqs_complex, self.args.device)

        # Cache the keys and values
        self.cache_k[:_bsz, start_pos : start_pos + _seqlen] = xk
        self.cache_v[:_bsz, start_pos : start_pos + _seqlen] = xv

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = output.transpose(1, 2).contiguous().view(_bsz, _seqlen, -1)
        return self.wo(output)  # (B, 1, Dim) -> (B, 1, Dim)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_complex: torch.Tensor, start_pos: int):
        h = x + self.attention.forward(self.attention_norm(x), freqs_complex, start_pos)
        h = h + self.feed_forward.forward(self.ffn_norm(h))
        return h


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, _seqlen = tokens.shape

        assert _seqlen == 1, "Only one token is allowed to be passed at a time"

        h = self.tok_embeddings(tokens)

        # Precompute the frequencies for the current position
        freqs_complex = self.freqs_complex[start_pos : start_pos + _seqlen]

        for layer in self.layers:
            h = layer(h, freqs_complex)

        h = self.norm(h)

        output = self.output(h).float()

        return output
