import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

class StamImu(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.adapter = Adapter(config)
        self.decoder = Decoder(config)

        self.pre_ln = RmsNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.fc_hidden)
        self.fc2 = nn.Linear(config.fc_hidden, config.output_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.adapter(x)
        x = self.decoder(x)
        x = self.pre_ln(x).mean(dim=1)
        x = F.silu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class StamImuStaticAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.adapter = AdapterStatic(config)
        self.decoder = Decoder(config)

        self.pre_ln = RmsNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.fc_hidden)
        self.fc2 = nn.Linear(config.fc_hidden, config.output_dim)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.adapter(x)
        x = self.decoder(x)
        x = self.pre_ln(x).mean(dim=1)
        x = F.silu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

class RmsNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, intermediate_size, dropout):
        super().__init__()
        self.norm1 = RmsNorm(hidden_size)
        self.norm2 = RmsNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True, dropout=dropout)
        self.ffn = SwiGluMlp(hidden_size, intermediate_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Linear(config.input_size, config.proj_size)
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(config.proj_size, config.n_heads, config.intermediate_size, config.dropout)
             for _ in range(config.e_layers)]
        )
        self.proj = nn.Linear(config.proj_size, config.adapter_hidden_size)
        self.norm = RmsNorm(config.proj_size)

    def forward(self, x):
        x = self.in_proj(x)
        for blk in self.encoder_layers:
            x = blk(x)
        x = self.proj(self.norm(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_conv, d_ff, expand):
        super().__init__()
        d_inner = int(d_model * expand)
        dt_rank = math.ceil(d_model / 16)

        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + d_ff * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        A = repeat(torch.arange(1, d_ff + 1), "n -> d n", d=d_inner).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = RmsNorm(d_model)

    def forward(self, x):
        _, l, _ = x.shape
        x_and_res = self.in_proj(self.norm(x))
        x, res = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        return self.out_proj(y)

    def ssm(self, x):
        _, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        delta, Bm, Cm = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, Bm, Cm, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")
        x = torch.zeros((b, d_in, n), device=deltaA.device, dtype=deltaA.dtype)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(config.d_model, config.d_conv, config.d_ff, config.expand)
            for _ in range(config.d_layers)
        ])

    def forward(self, x):
        for blk in self.layers:
            x = x + blk(x)
        return x

class AdapterBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.width = config.adapter_hidden_size
        self.num_heads = config.adapter_num_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = config.adapter_mini_batch_size

        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))

        self._init_qkvo_proj()
        self._init_lr_gate()
        self._init_ln()

        self.use_gate = config.adapter_use_gate
        if self.use_gate:
            self.g_proj = nn.Linear(self.width, self.width, bias=False)

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

    def _init_lr_gate(self):
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.learnable_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        self.learnable_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )
        return XQ, XK, XV

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def get_eta(self, X, mini_batch_size):
        lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_lr_weight) + self.learnable_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        lr = torch.sigmoid(lr)
        lr = lr.permute(0, 1, 2, 4, 3)
        lr_eta = self.config.adapter_base_lr * lr / self.head_dim

        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[: mini_batch_size]

        token_idx = torch.clamp_min(token_idx, 0.0)
        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )

        return token_eta, lr_eta

    def apply_gate(self, hidden_states, output):
        y = self.g_proj(hidden_states)
        y = F.gelu(y, approximate="tanh")
        output = y * output
        return output

    def forward(self, hidden_states: torch.Tensor):
        B, L = hidden_states.shape[:2]

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        output_hidden_states = []

        reminder_len = L % self.mini_batch_size
        num_mini_batch = L // self.mini_batch_size

        if num_mini_batch > 0:
            inputs = {
                "XQ": XQ[:, :, : num_mini_batch * self.mini_batch_size],
                "XK": XK[:, :, : num_mini_batch * self.mini_batch_size],
                "XV": XV[:, :, : num_mini_batch * self.mini_batch_size],
                "X": hidden_states[:, : num_mini_batch * self.mini_batch_size],
            }
            output_mod = self.process(
                self.get_inputs(inputs, self.mini_batch_size),
                mini_batch_size=self.mini_batch_size,
            )
            output_hidden_states.append(output_mod)

        if reminder_len > 0:
            inputs = {
                "XQ": XQ[:, :, -reminder_len:],
                "XK": XK[:, :, -reminder_len:],
                "XV": XV[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            output_reminder = self.process(
                self.get_inputs(inputs, reminder_len),
                mini_batch_size=reminder_len,
            )
            output_hidden_states.append(output_reminder)

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        if self.use_gate:
            output_hidden_states = self.apply_gate(hidden_states, output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)

        return output_hidden_states

    def get_inputs(self, inputs, mini_batch_size):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X  = inputs["X"]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        X  = X.reshape(B, num_mini_batch, mini_batch_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)

        token_eta, lr_eta = self.get_eta(X, mini_batch_size)
        eta = token_eta * lr_eta
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "lr_eta": lr_eta,
        }
        return inputs


class AdapterLayer(AdapterBase):
    def __init__(self, config):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def process(self, inputs, mini_batch_size):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict["W1_states"]
            b1_init = params_dict["b1_states"]
            W2_init = params_dict["W2_states"]
            b2_init = params_dict["b2_states"]

            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            eta_mini_batch = inputs["eta"]

            X1 = XK_mini_batch
            Z1 = X1 @ W1_init + b1_init
            X2 = F.gelu(Z1, approximate="tanh")
            Z2 = X2 @ W2_init + b2_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias  = self.norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z2 = layer_norm_l2_backward(Z2, reconstruction_target, ln_weight, ln_bias)
            grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

            Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
            b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
            Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
            X2_bar = F.gelu(Z1_bar, approximate="tanh")

            Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
            b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
            Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

            last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
            W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
            b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
            W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
            b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)

            Z2_bar = layer_norm(Z2_bar, ln_weight, ln_bias)
            XQW_mini_batch = XQ_mini_batch + Z2_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W2_states": W2_last,
                "b2_states": b2_last,
            }
            return last_param_dict, XQW_mini_batch

        init_params_dict = {
            "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)),
            "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1)),
        }

        def tree_map(fn, tree):
            if isinstance(tree, dict):
                return {k: fn(v) for k, v in tree.items()}
            return fn(tree)

        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        _, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
        )

        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch


class AdapterBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_modeling_block = AdapterLayer(config=config)
        self.mlp = SwiGluMlp(
            config.adapter_hidden_size,
            config.adapter_intermediate_size,
        )
        self.seq_norm = RmsNorm(config.adapter_hidden_size)
        self.ffn_norm = RmsNorm(config.adapter_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.seq_norm(hidden_states)
        hidden_states = self.seq_modeling_block(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AdapterModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AdapterBlock(config) for _ in range(config.adapter_num_layers)])
        self.norm = RmsNorm(config.adapter_hidden_size)

    def forward(self, x):
        hidden_states = x
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adapter = AdapterModule(config)
        hs = config.adapter_hidden_size
        d_model_out = getattr(config, "d_model", hs)
        self.out_proj = nn.Identity() if d_model_out == hs else nn.Linear(hs, d_model_out, bias=False)

    def forward(self, x):
        out = self.adapter(x)
        return self.out_proj(out)

class AdapterMlpStatic(AdapterBase):
    def __init__(self, config):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1, 4 * self.head_dim))
        self.b2 = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1, self.head_dim))

    def process(self, inputs, mini_batch_size):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        B, H, nb, m, D = XK.shape
        Z1 = torch.einsum("bhnkd,hdf->bhnkf", XK, self.W1) + self.b1
        X2 = F.gelu(Z1, approximate="tanh")

        Z2 = torch.einsum("bhnkf,hfd->bhnkd", X2, self.W2) + self.b2
        ln_weight = self.norm_weight.view(H, 1, 1, D)
        ln_bias   = self.norm_bias.view(H, 1, 1, D)
        Z2 = layer_norm(Z2, ln_weight, ln_bias)

        XQW = XQ + Z2
        XQW = XQW.permute(0, 2, 3, 1, 4).reshape(B, nb * m, H * D)
        return XQW


class AdapterBlockStatic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_modeling_block = AdapterMlpStatic(config=config)
        self.mlp = SwiGluMlp(
            config.adapter_hidden_size,
            config.adapter_intermediate_size,
        )
        self.seq_norm = RmsNorm(config.adapter_hidden_size)
        self.ffn_norm = RmsNorm(config.adapter_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.seq_norm(hidden_states)
        hidden_states = self.seq_modeling_block(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AdapterModuleStatic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AdapterBlockStatic(config) for _ in range(config.adapter_num_layers)])
        self.norm = RmsNorm(config.adapter_hidden_size)

    def forward(self,x):
        hidden_states = x
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class AdapterStatic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adapter = AdapterModuleStatic(config)
        hs = config.adapter_hidden_size
        d_model_out = getattr(config, "d_model", hs)
        self.out_proj = nn.Identity() if d_model_out == hs else nn.Linear(hs, d_model_out, bias=False)

    def forward(self, x):
        out = self.adapter(x)
        return self.out_proj(out)

class SwiGluMlp(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gated = F.gelu(self.gate_proj(x))
        down_proj = self.down_proj(gated * self.up_proj(x))
        return down_proj

def layer_norm(x, gamma, beta, eps: float = 1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    return gamma * x_hat + beta


def layer_norm_l2_backward(x, l2_target, gamma, beta, eps: float = 1e-6):
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (1.0 / D) * (
        D * grad_x_hat
        - grad_x_hat.sum(dim=-1, keepdim=True)
        - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
    ) / std
    return z


def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)

def scan(f, init, xs, out):
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    carry = scan_fn(carry, 0, num_items)
    return carry, out
