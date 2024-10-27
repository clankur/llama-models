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
    rope_max_timescale: int = model_args.rope_theta
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
    embed = jnp.from_dlpack(asdlpack(weights["tok_embeddings.weight"].float()))
    unembed = jnp.from_dlpack(asdlpack(weights["tok_embeddings.weight"].float()))

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
            "(n_q_per_kv n_kv d_head) d_model -> d_model n_q_per_kv n_kv d_head",
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
    tensor1: jax.Array | torch.Tensor,
    tensor2: jax.Array | torch.Tensor,
    tolerance: float = 2e-5,
) -> tuple[bool, bool]:
    # Convert the torch tensor to a jax array

    print(f"{tensor1.dtype=}, {tensor2.dtype=}")
    tensor1 = torch.from_dlpack(asdlpack(tensor1))
    tensor2 = torch.from_dlpack(asdlpack(tensor2))

    # Check if shapes are the same
    if tensor1.shape != tensor2.shape:
        return False, False

    # Check for exact match
    exact_match = torch.equal(tensor1, tensor2)

    # Check for approximate match
    max_diff = torch.max(torch.abs(tensor1 - tensor2))

    print(f"{ max_diff= }")
    approximate_match = max_diff <= tolerance

    return exact_match, approximate_match.item()


def rms_norm(x):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + h.norm_eps)


class RopeTable:
    def __init__(self, max_len: int, h: Hparams) -> None:
        head_dim = h.d_head
        timescale = 1.0 / (
            h.rope_max_timescale
            ** (jnp.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim)
        )
        position = jnp.arange(max_len, dtype=jnp.float32)
        # need to add scaling to 1.0/timescale here

        # sinusoid_inp = jnp.float32(position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
        sinusoid_inp = jnp.outer(position, timescale)

        self.sin = jnp.sin(sinusoid_inp)
        self.cos = jnp.cos(sinusoid_inp)
        self.freqs_cis = jnp.ones_like(sinusoid_inp) * (self.cos + 1j * self.sin)

    def apply(self, rearrange_spec, x, start_pos=0):
        x_complex = jnp.reshape(x.astype(jnp.float32), (*x.shape[:-1], -1, 2))

        x_complex = x_complex[0] + x_complex[1] * 1j
        freqs_cis = rearrange(self.freqs_cis, rearrange_spec)[
            :, start_pos : start_pos + x.shape[1], ...
        ]
        x_out = x_complex * freqs_cis
        x_out = jnp.stack([jnp.real(x_out), jnp.imag(x_out)], axis=-1).astype(x.dtype)
        return jnp.reshape(x_out, (*x_out.shape[:-2], -1)).astype(x)


# %%
L = 5
h = Hparams()
K_MASK = -2.3819763e38
# use_local_window_attn = False
# causal_mask = jnp.tril(jnp.ones((batch_size, L, L), dtype=jnp.bool_), 0)[
#     ..., jnp.newaxis, jnp.newaxis, :
# ]
# local_mask = jnp.triu(jnp.ones((batch_size, L, L), dtype=jnp.bool_), 1 - h.window_size)[
#     ..., jnp.newaxis, jnp.newaxis, :
# ]

# %%
batch_size = 1
seq_length = 5  # tokens.targets.shape[-1]
max_len = 2048
dummy_input = np.zeros((batch_size, seq_length))
jnp_dummy_input = dummy_input.astype(jnp.int32)
torch_dummy_input = torch.from_numpy(dummy_input).long()

rope_table = RopeTable(max_len * 2, h)
# %%
rope_table.freqs_cis.shape, model.freqs_cis.shape, compare_tensors(
    rope_table.freqs_cis.astype(jnp.complex64), model.freqs_cis
)
# %%
print("resetting freqs cis")
rope_table.freqs_cis = jnp.from_dlpack(asdlpack(model.freqs_cis))
rope_table.freqs_cis.shape, model.freqs_cis.shape, compare_tensors(
    rope_table.freqs_cis.astype(jnp.float32), model.freqs_cis.float()
)
# %%
output, intermediates = model.forward(torch_dummy_input, 0)
output, intermediates
# %%
i = 0
ids = jnp_dummy_input
x = embed[ids]
freqs_cis = rope_table.freqs_cis[:seq_length]
print(compare_tensors(x, intermediates["tracked_embed"][0]))

# def loop_body(carry, layer_weights):
w_q, w_kv, w_o, w_gate, w_up, w_down, ln1, ln2 = (
    attn_qs[0],
    attn_kvs[0],
    attn_os[0],
    mlp_gates[0],
    mlp_ups[0],
    mlp_downs[0],
    pre_attention_norms[0],
    pre_ffw_norms[0],
)

# %%
print(intermediates["nx"][0].shape)
nx = rms_norm(x) * ln1
print(compare_tensors(nx, intermediates["nx"][0].float()))

# %%
w_q_compare = rearrange(
    w_q, "n_kv n_q_per_kv d_head d_model -> (n_kv n_q_per_kv d_head) d_model"
)
print(
    compare_tensors(
        w_q_compare,
        model.layers[0].attention.wq.weight.detach().bfloat16(),
    ),
)
# %%
q = einsum(
    nx,
    w_q,  # torch.from_dlpack(asdlpack
    "B Qlen d_model, n_kv n_q_per_kv d_head d_model -> B Qlen n_kv n_q_per_kv d_head",
)
k, v = einsum(
    nx, w_kv, "B Klen d_model, k_v d_model n_kv d_head -> k_v B Klen n_kv d_head"
).astype(x)

q_compare = rearrange(
    q, "B Qlen n_kv n_q_per_kv d_head -> B Qlen (n_kv n_q_per_kv) d_head"
)
print(
    compare_tensors(
        q_compare,
        intermediates["xq"][0].float(),
    ),
)
print(
    compare_tensors(
        k,
        intermediates["xk"][0].float(),
    ),
)
# %%
q_torch = intermediates["xq"][0].float().contiguous()
q_jax = jnp.array(q_torch.numpy(), copy=True)
# %%
q_complex_jax = jnp.reshape(q_jax.astype(jnp.float32), (*q_jax.shape[:-1], -1, 2))
q_complex_jax = q_complex_jax[..., 0] + q_complex_jax[..., 1] * 1j
q_complex_torch = torch.view_as_complex(
    q_torch.float().reshape(*q_torch.shape[:-1], -1, 2)
)
q_complex_jax.shape, q_complex_torch.shape, compare_tensors(
    q_complex_jax, q_complex_torch
)
# %%
from model import reshape_for_broadcast

local_freqs_cis_torch = model.freqs_cis[:seq_length]
local_freqs_cis_jax = jnp.from_dlpack(asdlpack(local_freqs_cis_torch))
local_freqs_cis_torch = reshape_for_broadcast(local_freqs_cis_torch, q_complex_torch)
local_freqs_cis_jax = rearrange(local_freqs_cis_jax, "L d -> 1 L 1 d")
compare_tensors(q_complex_jax, q_complex_torch)
# %%
q_out_torch = torch.view_as_real(q_complex_torch * local_freqs_cis_torch).flatten(3)
q_out_jax = q_complex_jax * local_freqs_cis_jax
q_out_jax = jnp.stack([jnp.real(q_out_jax), jnp.imag(q_out_jax)], axis=-1).astype(
    x.dtype
)
q_out_jax = jnp.reshape(q_out_jax, (*q_out_jax.shape[:-2], -1)).astype(x)
compare_tensors(q_out_jax, q_out_torch)


# %%
q = rope_table.apply("L d -> 1 L 1 1 d", q)
k = rope_table.apply("L d -> 1 L 1 d", k)
# %%
q_compare = rearrange(
    q, "B Qlen n_kv n_q_per_kv d_head -> B Qlen (n_kv n_q_per_kv) d_head"
)
print(compare_tensors(q_compare, intermediates["xq_roped"][i]))
print(compare_tensors(k, intermediates["xk_roped"][i]))

q_preatt_scalar = h.d_head**-0.5
q_scaled = q * q_preatt_scalar

# %%
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

# %%
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
