# %%
import torch
import numpy as np
from jax.sharding import Mesh
import jax.numpy as jnp
import jax
from einops import rearrange, einsum
import json
import os
from importlib import reload
from typing import Optional
import model as llama_model
from model import Transformer, ModelArgs

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
)

# %%
model_path = "/Users/clankur/.llama/checkpoints/Llama3.2-1B/"
with open(f"{model_path}/params.json", "r") as f:
    params = json.load(f)

params

# %%
model_args = ModelArgs(**params)
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
    rope_max_timescale: int = model_args.rope_theta  # not sure
    norm_eps: float = model_args.norm_eps


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
    embed = jnp.array(
        weights["tok_embeddings.weight"].float().numpy(), dtype=jnp.float32
    )
    unembed = jnp.array(
        weights["tok_embeddings.weight"].float().numpy(), dtype=jnp.float32
    )

    # Loop through each layer to load weights
    for layer_id in range(16):  # Adjust the range to match your model layers
        # norms
        ln1 = jnp.array(
            weights[f"layers.{layer_id}.attention_norm.weight"].float().numpy(),
            dtype=jnp.float32,
        )
        ln2 = jnp.array(
            weights[f"layers.{layer_id}.ffn_norm.weight"].float().numpy(),
            dtype=jnp.float32,
        )
        pre_attention_norms.append(ln1)
        pre_ffw_norms.append(ln2)

        # attention weights
        w_q = jnp.array(
            weights[f"layers.{layer_id}.attention.wq.weight"].float().numpy(),
            dtype=jnp.float32,
        )

        w_q = rearrange(
            w_q,
            "(n_kv n_q_per_kv d_head) d_model -> d_model n_kv n_q_per_kv d_head",
            n_q_per_kv=h.n_q_per_kv,
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # d_model n_kv n_q_per_kv d_head

        attn_qs.append(w_q)

        w_k = jnp.array(
            weights[f"layers.{layer_id}.attention.wk.weight"].float().numpy(),
            dtype=jnp.float32,
        )

        # rearranging dims like n_kv can cause issues!
        w_k = rearrange(
            w_k,
            "(n_kv d_head) d_model -> d_model n_kv d_head",
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # M_dim n_kv H_dim

        w_v = jnp.array(
            weights[f"layers.{layer_id}.attention.wv.weight"].float().numpy(),
            dtype=jnp.float32,
        )

        w_v = rearrange(
            w_v,
            "(n_kv d_head) d_model -> d_model n_kv d_head",
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # M_dim n_kv H_dim
        w_kv = jnp.stack([w_k, w_v], axis=0)
        attn_kvs.append(w_kv)

        w_o = jnp.array(
            weights[f"layers.{layer_id}.attention.wo.weight"].float().numpy(),
            dtype=jnp.float32,
        )
        w_o = rearrange(
            w_o,
            "(n_q_per_kv n_kv d_head) d_model -> d_model n_q_per_kv n_kv d_head",
            n_q_per_kv=h.n_q_per_kv,
            n_kv=h.n_kv,
            d_head=h.d_head,
        )  # "d_model/d n_q_per_kv n_kv/t d_head"
        attn_os.append(w_o)

        # mlp
        w_gate = jnp.array(
            weights[f"layers.{layer_id}.feed_forward.w1.weight"].float().numpy(),
            dtype=jnp.float32,
        )
        w_gate = rearrange(w_gate, "d_ff d_model -> d_model d_ff")
        mlp_gates.append(w_gate)

        w_up = jnp.array(
            weights[f"layers.{layer_id}.feed_forward.w3.weight"].float().numpy(),
            dtype=jnp.float32,
        )
        w_up = rearrange(w_up, "d_ff d_model -> d_model d_ff")
        mlp_ups.append(w_up)

        w_down = jnp.array(
            weights[f"layers.{layer_id}.feed_forward.w2.weight"].float().numpy(),
            dtype=jnp.float32,
        )
        mlp_downs.append(w_down)

    # Stack all lists along the first dimension (number of layers)
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
) = load_llama(weights, h)


# %%
def compare_tensors(
    tensor1: jax.Array, tensor2: torch.Tensor, tolerance: float = 1e-5
) -> tuple[bool, bool]:
    # Convert the torch tensor to a jax array
    tensor2 = jnp.array(tensor2.cpu().numpy())

    # Check if shapes are the same
    if tensor1.shape != tensor2.shape:
        return False, False

    # Check for exact match
    exact_match = jnp.array_equal(tensor1, tensor2)

    # Check for approximate match
    max_diff = jnp.max(jnp.abs(tensor1 - tensor2))
    approximate_match = max_diff <= tolerance

    return exact_match, approximate_match


def rms_norm(x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + h.norm_eps)


class RopeTable:
    def __init__(self, max_len: int, h: Hparams) -> None:
        head_dim = h.d_head
        position = jnp.arange(max_len, dtype=jnp.int32)
        fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
        timescale = h.rope_max_timescale**fraction

        sinusoid_inp = jnp.float32(position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
        self.sin = jnp.sin(sinusoid_inp)
        self.cos = jnp.cos(sinusoid_inp)

    def apply(self, rearrange_spec, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = rearrange(self.sin, rearrange_spec)
        cos = rearrange(self.cos, rearrange_spec)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.concatenate([r1, r2], axis=-1).astype(x.dtype)


# %%
L = 5
h = Hparams()
K_MASK = -2.3819763e38
rope_table = RopeTable(seq_length, h)
use_local_window_attn = False
causal_mask = jnp.tril(jnp.ones((batch_size, L, L), dtype=jnp.bool_), 0)[
    ..., jnp.newaxis, jnp.newaxis, :
]
local_mask = jnp.triu(jnp.ones((batch_size, L, L), dtype=jnp.bool_), 1 - h.window_size)[
    ..., jnp.newaxis, jnp.newaxis, :
]

# %%

# %%
(
    embed,
    m_ln1,
    m_ln2,
    m_w_q,
    m_w_kv,
    m_w_o,
    m_post_attn_ln,
    m_w_gate,
    m_w_up,
    m_w_down,
    m_post_ffn_ln,
    m_final_layer_norm,
) = flatten_params_to_tensors(params, h)

# %%
i = 0
ids = dummy_input
x = embed[ids]
x *= jnp.sqrt(h.d_model)
print(compare_tensors(x, intermediates["tracked_embed"]))


def loop_body(carry, layer_weights):
    w_q, w_kv, w_o, w_gate, w_up, w_down, ln1, ln2, post_attn_ln, post_ffn_ln = (
        layer_weights
    )

    (x, use_local_window_attn, i) = carry

    jax.debug.print("layer {i} \n", i=i)
    print("initial carry dtype \n", x.dtype)

    nx = rms_norm(x) * (1.0 + ln1)
    jax.debug.print(
        "normed x alignment={b}",
        b=compare_tensors(nx, intermediates["pre_attention_norm"][i]),
    )

    # realigning
    # nx = intermediates['pre_attention_norm'][i]

    q = einsum(
        nx,
        w_q,
        "B Qlen d_model, d_model n_kv n_q_per_kv d_head -> B Qlen n_kv n_q_per_kv d_head",
    ).astype(x)
    k, v = einsum(
        nx, w_kv, "B Klen d_model, k_v d_model n_kv d_head -> k_v B Klen n_kv d_head"
    ).astype(x)

    q = rope_table.apply("L d -> 1 L 1 1 d", q)
    k = rope_table.apply("L d -> 1 L 1 d", k)
    q_preatt_scalar = h.d_head**-0.5
    q_scaled = q * q_preatt_scalar

    jax.debug.print(
        "roped_q alignment = {b}",
        b=compare_tensors(q_scaled, intermediates["reshaped_scaled_q"][i]),
    )
    jax.debug.print(
        "roped_k alignment = {b}", b=compare_tensors(k, intermediates["roped_k"][i])
    )

    logits = einsum(
        q_scaled,
        k,
        "B Qlen n_kv n_q_per_kv d_head, B Klen n_kv d_head -> B Qlen n_kv n_q_per_kv Klen",
    )
    logits = jnp.tanh(logits / h.attn_softcap) * h.attn_softcap
    logits_test = rearrange(
        logits, "B Qlen n_kv n_q_per_kv Klen -> B Qlen (n_kv n_q_per_kv) Klen"
    )
    jax.debug.print(
        "capped logits alignment={b}",
        b=compare_tensors(logits_test, intermediates["capped_logits"][i]),
    )

    attn_mask = jax.lax.select(
        use_local_window_attn,
        jnp.logical_and(causal_mask, local_mask),
        causal_mask,
    )
    logits = jnp.where(attn_mask, logits, -2.3819763e38)
    logits_test = rearrange(
        logits, "B Qlen n_kv n_q_per_kv Klen -> B Qlen (n_kv n_q_per_kv) Klen"
    )
    jax.debug.print(
        "masked logits alignment={b}",
        b=compare_tensors(logits_test, intermediates["masked_logits"][i]),
    )

    probs = jax.nn.softmax(logits, axis=-1).astype(x.dtype)
    probs_test = rearrange(
        probs, "B Qlen n_kv n_q_per_kv Klen -> B Qlen (n_kv n_q_per_kv) Klen"
    )
    jax.debug.print(
        "att wei alignment={b}",
        b=compare_tensors(probs_test, intermediates["att_wei"][i]),
    )

    encoded = einsum(
        probs,
        v,
        "B Qlen n_kv n_q_per_kv Klen, B Klen n_kv d_head -> B Qlen n_kv n_q_per_kv d_head",
    )

    encoded = rearrange(
        encoded, "B Qlen n_kv n_q_per_kv d_head -> B Qlen (n_kv n_q_per_kv) d_head"
    )

    # jax.debug.print("attn_out before MHA mix aligned: {b}", b=compare_tensors(
    #     encoded, intermediates['a_out_premix'][i]))

    # realigning
    encoded = intermediates["a_out_premix"][i]

    # for some reason: mixing mha is wrong here...
    # attn_out = einsum(
    #     encoded, w_o, "B Qlen n_head d_head, n_head d_head d_model  -> B Qlen d_model"
    # )
    # but correct here:
    attn_out = jnp.einsum("BTNH,NHD->BTD", encoded, w_o)

    jax.debug.print(
        "attn_out after w_o alignment: {b}",
        b=compare_tensors(attn_out, intermediates["a_out"][i]),
    )

    # realigning
    # attn_out = intermediates['a_out'][i]

    attn_out = rms_norm(attn_out) * (1.0 + post_attn_ln)
    jax.debug.print(
        "post attn norm alignment = {b}",
        b=compare_tensors(attn_out, intermediates["post_attention_norm"][i]),
    )

    # realigning
    attn_out = intermediates["post_attention_norm"][i]

    x += attn_out
    nx = rms_norm(x) * (1.0 + ln2)
    jax.debug.print(
        "pre ffw norm alignment = {b}",
        b=compare_tensors(nx, intermediates["pre_ffw_norm"][i]),
    )

    # realigning
    # nx = intermediates['pre_ffw_norm'][i]

    gate_proj = einsum(nx, w_gate, "B L M, M F -> B L F")
    up_proj = einsum(nx, w_up, "B L M, M F -> B L F")
    y = jax.nn.gelu(gate_proj) * up_proj
    ffn_out = einsum(y, w_down, "B L F, M F -> B L M")
    jax.debug.print(
        "ffn_out alignment = {b}", b=compare_tensors(ffn_out, intermediates["mlp"][i])
    )

    # realigning
    # ffn_out = intermediates['mlp'][i]
    ffn_out = rms_norm(ffn_out) * (1.0 + post_ffn_ln)
    jax.debug.print(
        "post_ffw_norm alignment = {b}",
        b=(compare_tensors(ffn_out, intermediates["post_ffw_norm"][i])),
    )

    # realigning
    ffn_out = intermediates["post_ffw_norm"][i]
    x += ffn_out

    print("final carry dtype \n", x.dtype)

    return (jnp.bfloat16(x), ~use_local_window_attn, i + 1), ()


for i in range(h.layers):
    layer_weights = [
        m_w_q[i],
        m_w_kv[i],
        m_w_o[i],
        m_w_gate[i],
        m_w_up[i],
        m_w_down[i],
        m_ln1[i],
        m_ln2[i],
        m_post_attn_ln[i],
        m_post_ffn_ln[i],
    ]
    (x, use_local_window_attn, i), _ = loop_body(
        (x, use_local_window_attn, i), layer_weights
    )

x = rms_norm(x) * (1.0 + m_final_layer_norm)
print(compare_tensors(x, intermediates["final_norm"]))
logits = einsum(x, embed, "B L M, V M ->B L V")
print(compare_tensors(logits, intermediates["tracked_unembed"]))
logits = jnp.tanh(logits / h.final_softcap) * h.final_softcap
print(compare_tensors(logits, intermediates["final_softcap"]))


# %%
i = 0
ids = dummy_input
x = embed[ids]
x *= jnp.sqrt(h.d_model)
(x, _, _), () = jax.lax.scan(
    loop_body,
    (x, False, i),
    (
        m_w_q,
        m_w_kv,
        m_w_o,
        m_w_gate,
        m_w_up,
        m_w_down,
        m_ln1,
        m_ln2,
        m_post_attn_ln,
        m_post_ffn_ln,
    ),
)
x = rms_norm(x) * (1.0 + m_final_layer_norm)
print(compare_tensors(x, intermediates["final_norm"]))
logits = einsum(x, embed, "B L M, V M ->B L V")
print(compare_tensors(logits, intermediates["tracked_unembed"]))
logits = jnp.tanh(logits / h.final_softcap) * h.final_softcap
print(compare_tensors(logits, intermediates["final_softcap"]))


# %%
