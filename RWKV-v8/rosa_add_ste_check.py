"""Check exact-forward/proxy-backward behavior in the addition ROSA branch."""

import torch

from rosa_add100_smoke import ROSAQKV, exact_rosa_targets


def main():
    torch.manual_seed(2026)
    channels = 8
    ste_gradient_scale = 0.5
    module = ROSAQKV(
        channels,
        dropout=0.0,
        sign_flip_p=0.0,
        ste_gradient_scale=ste_gradient_scale,
    ).eval()
    with torch.no_grad():
        module.output.weight.copy_(torch.eye(channels))
        module.output.bias.zero_()
        module.emb.fill_(1.0)

    x = torch.randn(2, 16, channels, requires_grad=True)
    captured = {}

    def capture_rosa_signal(_module, inputs):
        captured["signal"] = inputs[0].detach().clone()

    hook = module.output.register_forward_pre_hook(capture_rosa_signal)
    output, _unused_distill_loss = module(x)
    hook.remove()

    xx = module.time_shift(x) - x
    q = module.q(x + xx * module.x_q)
    k = module.k(x + xx * module.x_k)
    v = module.v(x + xx * module.x_v)
    target = exact_rosa_targets(q, k, v)
    exact_signal = 2 * target.to(output.dtype) - 1
    assert target.unique().numel() == 2, "expected both binary target classes in this fixture"
    assert torch.equal(captured["signal"], exact_signal)
    assert torch.equal(output, exact_signal)

    probe = torch.randn_like(output)
    task_only_loss = (output * probe).sum()
    task_only_loss.backward()
    task_input_grad = x.grad.detach().clone()
    task_parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if (name.startswith(("q.", "k.", "v.", "proxy.")) or name in {"x_q", "x_k", "x_v"})
        and parameter.grad is not None
    }
    assert task_parameter_grads, "task loss did not reach the ROSA proxy or q/k/v projections"
    assert torch.isfinite(task_input_grad).all()
    assert torch.isfinite(torch.stack([grad.norm() for grad in task_parameter_grads.values()])).all()
    assert task_input_grad.norm().item() > 0
    assert task_parameter_grads["proxy.output.weight"].norm().item() > 0

    # Compare with the same module using the soft proxy value as its direct
    # forward activation. STE should give the same task-loss gradient upstream.
    module.zero_grad(set_to_none=True)
    reference_x = x.detach().clone().requires_grad_(True)
    ref_xx = module.time_shift(reference_x) - reference_x
    ref_q = module.q(reference_x + ref_xx * module.x_q)
    ref_k = module.k(reference_x + ref_xx * module.x_k)
    ref_v = module.v(reference_x + ref_xx * module.x_v)
    ref_logits = module.proxy(ref_q, ref_k, ref_v)
    ref_proxy_probs = ref_logits.softmax(dim=-1)
    ref_proxy_signal = ref_proxy_probs[..., 1] - ref_proxy_probs[..., 0]
    reference_output = module.output(ref_proxy_signal * module.emb)
    (reference_output * probe).sum().backward()

    scaled_reference_input_grad = reference_x.grad * ste_gradient_scale
    input_gradient_error = (task_input_grad - scaled_reference_input_grad).abs().max().item()
    assert torch.allclose(task_input_grad, scaled_reference_input_grad, atol=1e-7, rtol=1e-6)
    gradient_errors = {}
    gradient_match = {}
    for name, parameter in module.named_parameters():
        if not (name.startswith(("q.", "k.", "v.", "proxy.")) or name in {"x_q", "x_k", "x_v"}):
            continue
        assert parameter.grad is not None, f"missing reference gradient: {name}"
        assert name in task_parameter_grads, f"missing STE gradient: {name}"
        scaled_reference_grad = parameter.grad * ste_gradient_scale
        gradient_errors[name] = (
            task_parameter_grads[name] - scaled_reference_grad
        ).abs().max().item()
        gradient_match[name] = torch.allclose(
            task_parameter_grads[name], scaled_reference_grad, atol=1e-7, rtol=1e-6
        )

    max_parameter_gradient_error = max(gradient_errors.values())
    assert all(gradient_match.values()), "STE gradients differ from the direct proxy path"
    print(
        "STE check passed: exact forward is bitwise equal; task-only backward matches "
        "the scaled direct proxy path upstream."
    )
    print(
        f"shape={tuple(output.shape)} ste_gradient_scale={ste_gradient_scale:g} "
        f"input_grad_norm={task_input_grad.norm().item():.6g} "
        f"proxy_head_grad_norm={task_parameter_grads['proxy.output.weight'].norm().item():.6g}"
    )
    print(
        f"max_input_gradient_error={input_gradient_error:.3e} "
        f"max_proxy_or_qkv_gradient_error={max_parameter_gradient_error:.3e}"
    )


if __name__ == "__main__":
    main()
