# %%
import torch
import numpy as np
import jax
from dlpack import asdlpack
from jax.sharding import Mesh
import jax.numpy as jnp
from einops import rearrange, einsum
import json
import os
from importlib import reload
from typing import Optional
import model as llama_model
from model import Transformer, ModelArgs, intermediates

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)

# %%
model_path = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/")
with open(f"{model_path}/params.json", "r") as f:
    params = json.load(f)
model_args = ModelArgs(**params)
params
# %%
# %%
reload(llama_model)
model = Transformer(model_args)
weights = torch.load(
    f"{model_path}/consolidated.00.pth", map_location=torch.device("cpu")
)
model.load_state_dict(weights)
weights
# %%
hidden_dim = 4 * model_args.dim
hidden_dim = int(2 * hidden_dim / 3)
hidden_dim = int(model_args.ffn_dim_multiplier * hidden_dim)
hidden_dim = model_args.multiple_of * (
    (hidden_dim + model_args.multiple_of - 1) // model_args.multiple_of
)


class Hparams:
    d_model: int = model_args.dim
    n_q_per_kv: int = model_args.n_heads // model_args.n_kv_heads
    n_kv: int = model_args.n_kv_heads
    d_head: int = model_args.dim // model_args.n_heads
    layers: int = model_args.n_layers
    vocab: int = model_args.vocab_size
    d_ff: int = hidden_dim
    rope_max_timescale: int = model_args.rope_theta
    norm_eps: float = model_args.norm_eps
    use_scale: bool = model_args.use_scaled_rope


# %%

h = Hparams
# %%


def load_llama(weights, h: Hparams):
    pre_attention_norms = []
    pre_ffw_norms = []
    attn_qs = []
    attn_kvs = []
    attn_os = []
    mlp_gates = []
    mlp_ups = []
    mlp_downs = []

    # Convert weights to JAX arrays
    embed = jnp.from_dlpack(asdlpack(weights["tok_embeddings.weight"].float()))
    unembed = jnp.from_dlpack(asdlpack(weights["tok_embeddings.weight"].float()))
    final_norm = jnp.from_dlpack(asdlpack(weights["norm.weight"].float()))
    # Loop through each layer to load weights
    for layer_id in range(16):  # Adjust the range to match your model layers
        # norms
        ln1 = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention_norm.weight"].float())
        )
        ln2 = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.ffn_norm.weight"].float())
        )
        pre_attention_norms.append(ln1)
        pre_ffw_norms.append(ln2)

        # attention weights
        w_q = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention.wq.weight"].float()),
        )

        w_q = rearrange(
            w_q,
            "(n_kv n_q_per_kv d_head) d_model  -> n_kv n_q_per_kv d_head d_model",
            n_q_per_kv=h.n_q_per_kv,
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # d_model n_kv n_q_per_kv d_head

        attn_qs.append(w_q)

        w_k = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention.wk.weight"].float()),
        )

        # rearranging dims like n_kv can cause issues!
        w_k = rearrange(
            w_k,
            "(n_kv d_head) d_model -> d_model n_kv d_head",
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # M_dim n_kv H_dim

        w_v = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention.wv.weight"].float()),
        )

        w_v = rearrange(
            w_v,
            "(n_kv d_head) d_model -> d_model n_kv d_head",
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # M_dim n_kv H_dim
        w_kv = jnp.stack([w_k, w_v], axis=0)
        attn_kvs.append(w_kv)

        w_o = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.attention.wo.weight"].float()),
        )
        w_o = rearrange(
            w_o,
            "d_model (n_kv n_q_per_kv d_head) -> d_model n_kv n_q_per_kv d_head",
            n_q_per_kv=h.n_q_per_kv,
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # "d_model/d n_q_per_kv n_kv/t d_head"
        attn_os.append(w_o)

        # mlp
        w_gate = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.feed_forward.w1.weight"].float()),
        )
        w_gate = rearrange(w_gate, "d_ff d_model -> d_model d_ff")
        mlp_gates.append(w_gate)

        w_up = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.feed_forward.w3.weight"].float()),
        )
        w_up = rearrange(w_up, "d_ff d_model -> d_model d_ff")
        mlp_ups.append(w_up)

        w_down = jnp.from_dlpack(
            asdlpack(weights[f"layers.{layer_id}.feed_forward.w2.weight"].float()),
        )
        mlp_downs.append(w_down)

    pre_attention_norms = jnp.stack(pre_attention_norms, axis=0)
    pre_ffw_norms = jnp.stack(pre_ffw_norms, axis=0)
    attn_qs = jnp.stack(attn_qs, axis=0)
    attn_kvs = jnp.stack(attn_kvs, axis=0)
    attn_os = jnp.stack(attn_os, axis=0)
    mlp_gates = jnp.stack(mlp_gates, axis=0)
    mlp_ups = jnp.stack(mlp_ups, axis=0)
    mlp_downs = jnp.stack(mlp_downs, axis=0)

    return (
        embed,
        unembed,
        pre_attention_norms,
        pre_ffw_norms,
        attn_qs,
        attn_kvs,
        attn_os,
        mlp_gates,
        mlp_ups,
        mlp_downs,
        final_norm,
    )


# %%
(
    embed,
    unembed,
    pre_attention_norms,
    pre_ffw_norms,
    attn_qs,
    attn_kvs,
    attn_os,
    mlp_gates,
    mlp_ups,
    mlp_downs,
    final_norm,
) = load_llama(weights, h)


# %%
def compare_tensors(
    tensor1: jax.Array | torch.Tensor,
    tensor2: jax.Array | torch.Tensor,
    tolerance: float = 1e-4,
) -> tuple[bool, bool]:
    tensor1 = torch.from_dlpack(asdlpack(tensor1))
    tensor2 = torch.from_dlpack(asdlpack(tensor2))

    # Check if shapes are the same
    if tensor1.shape != tensor2.shape:
        print(f"tensors don't have same shape: {tensor1.shape=}, {tensor2.shape=}")
        return False, False

    # Check for exact match
    exact_match = torch.equal(tensor1, tensor2)

    # Check for approximate match
    max_diff = torch.max(torch.abs(tensor1 - tensor2))

    approximate_match = max_diff <= tolerance

    return exact_match, approximate_match.item(), f"{max_diff=}"


def rms_norm(x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + h.norm_eps)


class RopeTable:
    def __init__(self, max_len: int, h: Hparams) -> None:
        head_dim = h.d_head
        fraction = jnp.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim
        timescale = h.rope_max_timescale**fraction
        position = jnp.arange(max_len, dtype=jnp.float32)

        if h.use_scale:
            timescale = self._apply_scaling_factor(1.0 / timescale)
        sinusoid_inp = jnp.outer(position, timescale)

        self.sin = jnp.sin(sinusoid_inp)
        self.cos = jnp.cos(sinusoid_inp)
        self.freqs_cis = jnp.ones_like(sinusoid_inp) * (self.cos + 1j * self.sin)

    def _apply_scaling_factor(self, freqs):
        scale_factor = 8
        low_freq_factor = 1
        high_freq_factor = 4
        old_context_len = 8192  # original llama3 length

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        new_freqs = []
        for freq in freqs:
            wavelen = 2 * jnp.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return jnp.array(new_freqs, dtype=freqs.dtype)

    def apply(self, rearrange_spec, x, start_pos=0):
        x_complex = jnp.reshape(x.astype(jnp.float32), (*x.shape[:-1], -1, 2))

        x_complex = x_complex[..., 0] + x_complex[..., 1] * 1j
        freqs_cis = rearrange(
            self.freqs_cis[start_pos : start_pos + x.shape[1]], rearrange_spec
        )

        x_out = x_complex * freqs_cis
        x_out = jnp.stack([jnp.real(x_out), jnp.imag(x_out)], axis=-1).astype(x.dtype)
        return jnp.reshape(x_out, (*x_out.shape[:-2], -1)).astype(x)


# %%
batch_size = 1
seq_length = 5  # tokens.targets.shape[-1]
max_len = 2048
L = 5
h = Hparams()
K_MASK = -2.3819763e38
causal_mask = jnp.tril(jnp.ones((batch_size, L, L), dtype=jnp.bool_), 0)[
    ..., jnp.newaxis, jnp.newaxis, :
]

# %%
dummy_input = np.zeros((batch_size, seq_length))
jnp_dummy_input = dummy_input.astype(jnp.int32)
torch_dummy_input = torch.from_numpy(dummy_input).long()
rope_table = RopeTable(max_len * 2, h)

# %%
rope_table.freqs_cis.shape, model.freqs_cis.shape, compare_tensors(
    rope_table.freqs_cis.astype(jnp.complex64), model.freqs_cis
)
# %%
output, intermediates = model.forward(torch_dummy_input, 0)
output, intermediates


# %%
def loop_body(carry, layer_weights):
    x, i = carry
    w_q, w_kv, w_o, w_gate, w_up, w_down, ln1, ln2 = layer_weights
    nx = rms_norm(x) * ln1
    q = einsum(
        nx,
        w_q,
        "B Qlen d_model, n_kv n_q_per_kv d_head d_model -> B Qlen n_kv n_q_per_kv d_head",
    )
    k, v = einsum(
        nx, w_kv, "B Klen d_model, k_v d_model n_kv d_head -> k_v B Klen n_kv d_head"
    ).astype(x)
    q_compare = rearrange(
        q, "B Qlen n_kv n_q_per_kv d_head -> B Qlen (n_kv n_q_per_kv) d_head"
    )

    q = rope_table.apply("L d -> 1 L 1 1 d", q)
    k = rope_table.apply("L d -> 1 L 1 d", k)
    q_compare = rearrange(
        q, "B Qlen n_kv n_q_per_kv d_head -> B Qlen (n_kv n_q_per_kv) d_head"
    )

    q_preatt_scalar = h.d_head**-0.5
    q_scaled = q * q_preatt_scalar

    logits = einsum(
        q_scaled,
        k,
        "B Qlen n_kv n_q_per_kv d_head, B Klen n_kv d_head -> B Qlen n_kv n_q_per_kv Klen",
    )
    logits_test = rearrange(
        logits, "B Qlen n_kv n_q_per_kv Klen -> B (n_kv n_q_per_kv) Qlen Klen"
    )

    logits = jnp.where(causal_mask, logits, K_MASK)
    probs = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
    probs_test = rearrange(
        probs, "B Qlen n_kv n_q_per_kv Klen -> B (n_kv n_q_per_kv) Qlen Klen"
    )

    attn_out = einsum(
        probs,
        v,
        "B Qlen n_kv n_q_per_kv Klen, B Klen n_kv d_head -> B Qlen n_kv n_q_per_kv d_head",
    )

    attn_out_prime = einsum(
        attn_out,
        w_o,
        "B Qlen n_kv n_q_per_kv d_head, d_model n_kv n_q_per_kv d_head -> B Qlen d_model ",
    )
    x += attn_out_prime

    nx = rms_norm(x) * ln2

    gate_proj = einsum(nx, w_gate, "B L M, M F -> B L F")
    up_proj = jnp.einsum("blm,mf->blf", nx, w_up, precision=jax.lax.Precision.HIGHEST)

    y = jax.nn.silu(gate_proj) * up_proj
    ffn_out = jnp.einsum("blf,mf->blm", y, w_down, precision=jax.lax.Precision.HIGHEST)
    x += ffn_out

    return (x, i), ()


# %%
i = 0
ids = jnp_dummy_input
x = embed[ids]
freqs_cis = rope_table.freqs_cis[:seq_length]
print(compare_tensors(x, intermediates["tracked_embed"][i]))
(x, i), () = jax.lax.scan(
    loop_body,
    (x, i),
    (
        attn_qs,
        attn_kvs,
        attn_os,
        mlp_gates,
        mlp_ups,
        mlp_downs,
        pre_attention_norms,
        pre_ffw_norms,
    ),
)
x = rms_norm(x) * (final_norm)
print(compare_tensors(x, intermediates["final_norm"][0]))
logits = einsum(x, embed, "B L M, V M ->B L V")
print(compare_tensors(logits, intermediates["tracked_unembed"][0]))
# %%
