import math
import torch
import torch.nn.functional as F


def sft_loss(output):
    if output.loss is None or not torch.isfinite(output.loss):
        raise ValueError('Invalid SFT loss')
    return output.loss


def response_logprob(logits, labels, chunk_tokens=16):
    shifted = labels[:, 1:]
    positions = shifted.ne(-100).nonzero(as_tuple=False)
    if len(positions) == 0:
        raise ValueError('Response has no supervised tokens')
    result = logits.new_zeros((), dtype=torch.float32)
    for chunk in positions.split(chunk_tokens):
        scores = logits[chunk[:, 0], chunk[:, 1], :].float()
        ids = shifted[chunk[:, 0], chunk[:, 1]]
        result = result + F.log_softmax(scores, dim=-1).gather(1, ids[:, None]).sum()
    return result, len(positions)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    return -F.logsigmoid(beta * ((chosen - rejected) - (ref_chosen - ref_rejected)))


def dpo_gradient_coefficient(policy_margin, reference_margin, beta):
    # Derivative with respect to policy chosen-minus-rejected log probability.
    z = beta * (policy_margin - reference_margin)
    return -beta * math.exp(-F.softplus(torch.tensor(z, dtype=torch.float64)).item())
