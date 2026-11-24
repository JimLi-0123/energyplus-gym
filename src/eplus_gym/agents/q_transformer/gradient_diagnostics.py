"""Measure task gradients without changing optimizer gradients or loss weights."""
import math
import torch


@torch.no_grad()
def occupancy_prediction_metrics(prediction, target, current):
    """Normalized errors on the same replay batch; persistence predicts current occupancy."""
    prediction, target, current = (t.detach().reshape(-1) for t in (prediction, target, current))
    if prediction.shape != target.shape or target.shape != current.shape:
        raise ValueError('Occupancy prediction, target and current must have matching shapes.')
    changed = ~torch.isclose(target, current, rtol=0, atol=1e-7)
    prediction_error = (prediction - target).square()
    persistence_error = (current - target).square()
    total, count = target.numel(), int(changed.sum().item())
    result = {'occ_sample_count': total, 'occ_changed_count': count}
    for name, errors in (('prediction', prediction_error), ('persistence', persistence_error)):
        result['occ_' + name + '_sse'] = errors.sum().item()
        result['occ_' + name + '_mse'] = errors.mean().item()
        result['occ_changed_' + name + '_sse'] = errors[changed].sum().item()
        result['occ_changed_' + name + '_mse'] = errors[changed].mean().item() if count else float('nan')
    return result


def compare_loss_gradients(q_loss, aux_loss, parameters):
    parameters = tuple(p for p in parameters if p.requires_grad)
    q_grads = torch.autograd.grad(q_loss, parameters, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(aux_loss, parameters, retain_graph=True, allow_unused=True)
    device = q_loss.device
    q_squared = torch.zeros((), dtype=torch.float64, device=device)
    aux_squared = torch.zeros_like(q_squared)
    dot = torch.zeros_like(q_squared)
    for q_grad, aux_grad in zip(q_grads, aux_grads):
        if q_grad is not None:
            q_grad = q_grad.detach().double()
            q_squared += q_grad.square().sum()
        if aux_grad is not None:
            aux_grad = aux_grad.detach().double()
            aux_squared += aux_grad.square().sum()
        if q_grad is not None and aux_grad is not None:
            dot += (q_grad * aux_grad).sum()
    q_norm, aux_norm = q_squared.sqrt().item(), aux_squared.sqrt().item()
    finite = math.isfinite(q_norm) and math.isfinite(aux_norm)
    valid = finite and q_norm > 0 and aux_norm > 0
    cosine = max(-1.0, min(1.0, dot.item() / (q_norm * aux_norm))) if valid else float('nan')
    return {
        'q_encoder_grad_norm': q_norm,
        'weighted_aux_encoder_grad_norm': aux_norm,
        'aux_to_q_grad_ratio': aux_norm / q_norm if finite and q_norm > 0 else float('nan'),
        'q_aux_grad_cosine': cosine,
        'gradient_direction_valid': valid,
        'gradient_conflict': cosine < 0 if valid else None,
    }
