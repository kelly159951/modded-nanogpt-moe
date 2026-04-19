import math

import torch


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile
def coupled_newtonschulz5_B(M_B, A, steps):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M_B.to(torch.bfloat16)
    A = A.to(torch.bfloat16)

    if len(M_B.shape) == 2:
        X = X / ((A @ X).norm() + 1e-7)
        for _ in range(steps):
            W = A @ X
            WTW = W.T @ W
            T = b * WTW + c * (WTW @ WTW)
            X = a * X + X @ T
    else:
        W = A @ X
        X = X / (W.norm(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + 1e-7)
        ATA = A.transpose(-1, -2) @ A
        for _ in range(steps):
            XXT = X @ X.transpose(-1, -2)
            MMATA = XXT @ ATA
            B = b * MMATA + c * (MMATA @ MMATA)
            X = a * X + B @ X
    return X.bfloat16()


@torch.compile
def coupled_newtonschulz5_A(M_A, B, steps):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M_A.to(torch.bfloat16)
    B = B.to(torch.bfloat16)

    if len(M_A.shape) == 2:
        X = X / ((X @ B).norm() + 1e-7)
        for _ in range(steps):
            W = X @ B
            WWT = W @ W.T
            T = b * WWT + c * (WWT @ WWT)
            X = a * X + T @ X
    else:
        W = X @ B
        X = X / (W.norm(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + 1e-7)
        BBT = B @ B.transpose(-1, -2)
        for _ in range(steps):
            XTX = X.transpose(-1, -2) @ X
            BBTMM = BBT @ XTX
            B_term = b * BBTMM + c * BBTMM @ BBTMM
            X = a * X + X @ B_term
    return X.bfloat16()


def _finalize_coupled_groups(block_module, coupled_pairs):
    paired_param_ids = set()
    for param_A, param_B, _, _ in coupled_pairs:
        if param_A.ndim != 2 or param_B.ndim != 2:
            raise ValueError("CoupledMuon only supports 2D paired parameters")
        if id(param_A) == id(param_B):
            raise ValueError("coupled pair cannot contain the same parameter twice")
        if id(param_A) in paired_param_ids or id(param_B) in paired_param_ids:
            raise ValueError("parameter appears in more than one coupled pair")
        paired_param_ids.add(id(param_A))
        paired_param_ids.add(id(param_B))

    muon_params = []
    adamw_params = [p for p in block_module.parameters() if id(p) not in paired_param_ids]
    return coupled_pairs, muon_params, adamw_params


def build_dense_coupled_muon_param_groups(raw_model):
    coupled_pairs = []
    for block in raw_model.transformer.h:
        coupled_pairs.append((block.attn.c_q.weight, block.attn.c_k.weight, raw_model.config.n_head, True))
        coupled_pairs.append((block.attn.c_proj.weight, block.attn.c_v.weight, raw_model.config.n_head, False))
        coupled_pairs.append((block.mlp.c_proj.weight, block.mlp.c_fc.weight, 1, False))
    return _finalize_coupled_groups(raw_model.transformer.h, coupled_pairs)


def build_moe_coupled_muon_param_groups(raw_model):
    coupled_pairs = []
    for block in raw_model.transformer.h:
        coupled_pairs.append((block.attn.c_q.weight, block.attn.c_k.weight, raw_model.config.n_head, True))
        coupled_pairs.append((block.attn.c_proj.weight, block.attn.c_v.weight, raw_model.config.n_head, False))
        for expert in block.mlp.experts:
            coupled_pairs.append((expert.c_proj.weight, expert.c_fc.weight, 1, False))
    return _finalize_coupled_groups(raw_model.transformer.h, coupled_pairs)


class CoupledMuon(torch.optim.Optimizer):
    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        coupled_pairs=None,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        coupled_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        use_multi_head=False,
        n_heads=1,
    ):
        coupled_pairs = coupled_pairs or []
        muon_params = muon_params or []
        adamw_params = adamw_params or []

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        all_params = []
        for param_A, param_B, _, _ in coupled_pairs:
            all_params.extend([param_A, param_B])
        all_params.extend(muon_params)
        all_params.extend(adamw_params)

        super().__init__(all_params, defaults)

        self.coupled_pairs = coupled_pairs
        self.coupled_steps = coupled_steps
        self.use_multi_head = use_multi_head
        self.n_heads = n_heads
        self.iter_num = 0

        for param_A, param_B, n_heads, is_qk in coupled_pairs:
            assert param_A.ndim == 2 and param_B.ndim == 2
            self.state[param_A]["use_coupled"] = True
            self.state[param_A]["coupled_with"] = param_B
            self.state[param_A]["is_A"] = True
            self.state[param_A]["is_qk"] = is_qk
            if isinstance(n_heads, tuple):
                self.state[param_A]["num_heads"] = n_heads[0]
                self.state[param_B]["num_heads"] = n_heads[1]
            else:
                self.state[param_A]["num_heads"] = n_heads
                self.state[param_B]["num_heads"] = n_heads
            self.state[param_B]["use_coupled"] = True
            self.state[param_B]["coupled_with"] = param_A
            self.state[param_B]["is_A"] = False
            self.state[param_B]["is_qk"] = is_qk

        for p in muon_params:
            assert p.ndim == 2
            self.state[p]["use_muon"] = True
            self.state[p]["use_coupled"] = False

        for p in adamw_params:
            self.state[p]["use_coupled"] = False
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            processed = set()
            for param_A, param_B, _, is_qk in self.coupled_pairs:
                if param_A in processed or param_B in processed:
                    continue

                g_A = param_A.grad
                g_B = param_B.grad
                if g_A is None or g_B is None:
                    continue

                state_A = self.state[param_A]
                state_B = self.state[param_B]

                if "momentum_buffer" not in state_A:
                    state_A["momentum_buffer"] = torch.zeros_like(g_A)
                if "momentum_buffer" not in state_B:
                    state_B["momentum_buffer"] = torch.zeros_like(g_B)

                buf_A = state_A["momentum_buffer"]
                buf_B = state_B["momentum_buffer"]

                buf_A.mul_(momentum).add_(g_A)
                buf_B.mul_(momentum).add_(g_B)

                if group["nesterov"]:
                    g_A_eff = g_A.add(buf_A, alpha=momentum)
                    g_B_eff = g_B.add(buf_B, alpha=momentum)
                else:
                    g_A_eff = buf_A
                    g_B_eff = buf_B

                if self.use_multi_head and param_A.shape[1] % state_A["num_heads"] == 0 and param_B.shape[0] % state_B["num_heads"] == 0:
                    if is_qk:
                        HD, I = param_A.shape
                        HD_B, I_B = param_B.shape
                        head_dim = HD_B // state_B["num_heads"]
                        if HD == HD_B:
                            g_A_reshaped = g_A_eff.view(state_A["num_heads"], 2, head_dim // 2, I).permute(0, 2, 3, 1).contiguous()
                            g_B_reshaped = g_B_eff.view(state_B["num_heads"], 2, head_dim // 2, I).permute(0, 2, 1, 3).contiguous()
                            p_A_reshaped = param_A.data.view(state_A["num_heads"], 2, head_dim // 2, I).permute(0, 2, 3, 1).contiguous()
                            p_B_reshaped = param_B.data.view(state_B["num_heads"], 2, head_dim // 2, I).permute(0, 2, 1, 3).contiguous()

                            u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps)
                            u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps)

                            u_A_c = u_A_c.permute(0, 3, 1, 2).contiguous().reshape(HD, I)
                            u_B_c = u_B_c.permute(0, 2, 1, 3).contiguous().reshape(HD_B, I_B)
                        else:
                            if HD > HD_B:
                                n_heads_group = HD // HD_B
                                g_A_reshaped = g_A_eff.view(n_heads_group, state_B["num_heads"], 2, head_dim // 2, I).permute(0, 1, 3, 4, 2).contiguous()
                                g_B_reshaped = g_B_eff.view(1, state_B["num_heads"], 2, head_dim // 2, I).permute(0, 1, 3, 2, 4).contiguous()
                                p_A_reshaped = param_A.data.view(n_heads_group, state_B["num_heads"], 2, head_dim // 2, I).permute(0, 1, 3, 4, 2).contiguous()
                                p_B_reshaped = param_B.data.view(1, state_B["num_heads"], 2, head_dim // 2, I).permute(0, 1, 3, 2, 4).contiguous()

                                u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps)
                                u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps)

                                u_A_c = u_A_c.permute(0, 1, 4, 2, 3).contiguous().reshape(HD, I)
                                u_B_c = u_B_c.mean(dim=0).permute(0, 2, 1, 3).contiguous().reshape(HD_B, I_B)
                            else:
                                n_heads_group = HD_B // HD
                                g_A_reshaped = g_A_eff.view(1, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 4, 2).contiguous()
                                g_B_reshaped = g_B_eff.view(n_heads_group, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 2, 4).contiguous()
                                p_A_reshaped = param_A.data.view(1, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 4, 2).contiguous()
                                p_B_reshaped = param_B.data.view(n_heads_group, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 2, 4).contiguous()

                                u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps)
                                u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps)

                                u_A_c = u_A_c.mean(dim=0).permute(0, 3, 1, 2).contiguous().reshape(HD, I)
                                u_B_c = u_B_c.permute(0, 1, 3, 2, 4).contiguous().reshape(HD_B, I_B)
                    else:
                        O, HD = param_A.shape
                        HD_B, I = param_B.shape
                        head_dim_A = HD // state_A["num_heads"]
                        head_dim_B = HD_B // state_B["num_heads"]

                        if state_A["num_heads"] == state_B["num_heads"]:
                            g_A_reshaped = g_A_eff.view(O, state_A["num_heads"], head_dim_A).permute(1, 0, 2)
                            g_B_reshaped = g_B_eff.view(state_B["num_heads"], head_dim_B, I)
                            p_A_reshaped = param_A.data.view(O, state_A["num_heads"], head_dim_A).permute(1, 0, 2)
                            p_B_reshaped = param_B.data.view(state_B["num_heads"], head_dim_B, I)

                            u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps)
                            u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps)

                            u_A_c = u_A_c.permute(1, 0, 2).reshape(O, HD)
                            u_B_c = u_B_c.reshape(HD_B, I)
                        elif state_A["num_heads"] > state_B["num_heads"]:
                            assert state_A["num_heads"] % state_B["num_heads"] == 0
                            n_repeats = state_A["num_heads"] // state_B["num_heads"]

                            g_A_reshaped = g_A_eff.view(O, state_A["num_heads"], head_dim_A).permute(1, 0, 2)
                            p_A_reshaped = param_A.data.view(O, state_A["num_heads"], head_dim_A).permute(1, 0, 2)
                            g_B_reshaped = g_B_eff.view(state_B["num_heads"], head_dim_B, I).repeat_interleave(n_repeats, dim=0)
                            p_B_reshaped = param_B.data.view(state_B["num_heads"], head_dim_B, I).repeat_interleave(n_repeats, dim=0)

                            u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps)
                            u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps)

                            u_A_c = u_A_c.permute(1, 0, 2).reshape(O, HD)
                            u_B_c = u_B_c.view(state_B["num_heads"], n_repeats, head_dim_B, I).mean(dim=1).reshape(HD_B, I)
                        else:
                            assert state_B["num_heads"] % state_A["num_heads"] == 0
                            n_repeats = state_B["num_heads"] // state_A["num_heads"]

                            g_A_reshaped = g_A_eff.view(O, state_A["num_heads"], head_dim_A).permute(1, 0, 2).repeat_interleave(n_repeats, dim=0)
                            p_A_reshaped = param_A.data.view(O, state_A["num_heads"], head_dim_A).permute(1, 0, 2).repeat_interleave(n_repeats, dim=0)
                            g_B_reshaped = g_B_eff.view(state_B["num_heads"], head_dim_B, I)
                            p_B_reshaped = param_B.data.view(state_B["num_heads"], head_dim_B, I)

                            u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps)
                            u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps)

                            u_A_c = u_A_c.view(state_A["num_heads"], n_repeats, O, head_dim_A).mean(dim=1).permute(1, 0, 2).reshape(O, HD)
                            u_B_c = u_B_c.reshape(HD_B, I)
                else:
                    u_A_c = coupled_newtonschulz5_A(g_A_eff, param_B.data, steps=self.coupled_steps)
                    u_B_c = coupled_newtonschulz5_B(g_B_eff, param_A.data, steps=self.coupled_steps)

                u_A = zeropower_via_newtonschulz5(u_A_c, steps=ns_steps)
                u_B = zeropower_via_newtonschulz5(u_B_c, steps=ns_steps)

                adjusted_lr_A = self.adjust_lr_for_muon(lr, param_A.shape)
                adjusted_lr_B = self.adjust_lr_for_muon(lr, param_B.shape)
                param_A.data.mul_(1 - lr * wd).add_(u_A, alpha=-adjusted_lr_A)
                param_B.data.mul_(1 - lr * wd).add_(u_B, alpha=-adjusted_lr_B)

                processed.add(param_A)
                processed.add(param_B)

            params = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                u = zeropower_via_newtonschulz5(g, steps=ns_steps)
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd).add_(u, alpha=-adjusted_lr)

            params = [p for p in group["params"] if not self.state[p].get("use_coupled", False) and not self.state[p].get("use_muon", False)]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.data.mul_(1 - lr * wd).add_(g, alpha=-lr / scale)

        self.iter_num += 1
        return loss
