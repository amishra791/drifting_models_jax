import jax.numpy as jnp
from flax import nnx
from jax import lax
import jax.nn as nn


def cdist(x, y):
    """
    Computes pairwise distances
    
    :param x: [N_x, D]
    :param y: [N_y, D]

    Returns a [N_x, N_y] mat of pairwise distances
    """
    return jnp.sqrt(jnp.sum((x[:, None, :] - y[None, ...]) ** 2, axis=-1))


def compute_drift_field(x_bd, ypos_bd, yneg_bd, temp: float = 0.05, eps: float = 1e-12):
    """
    Computes the drift field V_pq(f(eps))
    
    :param x_bd: [N, D]
    :param ypos_bd: p, distribution of the data. [N_pos, D]
    :param yneg_bd: q, current distribution. [N_neg, D]

    Note that the batch dimensions of the data and generated predictions
    do not have to be the same.

    Returns a [N, D] matrix representing a drift field.
    """

    targets = jnp.concatenate([yneg_bd, ypos_bd], axis=0)
    N_neg = x_bd.shape[0]

    dist = cdist(x_bd, targets)
    # since x_bd is the same as yneg_bd, mask self
    dist = dist.at[:, :N_neg].add(jnp.eye(N_neg) * 1e6)
    kernel = jnp.exp(-dist / temp)

    normalizer = jnp.sum(kernel, axis=-1, keepdims=True) * jnp.sum(kernel, axis=-2, keepdims=True)
    normalizer = jnp.sqrt(jnp.clip(normalizer, a_min=eps))
    normalized_kernel = kernel / normalizer

    K_neg, K_pos = jnp.split(normalized_kernel, [N_neg,], axis=1)

    pos_coeff = K_pos * jnp.sum(K_neg, axis=-1, keepdims=True)
    V_pos = pos_coeff @ ypos_bd
    neg_coeff = K_neg * jnp.sum(K_pos, axis=-1, keepdims=True)
    V_neg = neg_coeff @ yneg_bd

    return V_pos - V_neg


def drift_loss(model, z_bd, datasample_bd):
    gen_bd = model(z_bd)

    gen_sg = lax.stop_gradient(gen_bd)
    drift = compute_drift_field(gen_sg, datasample_bd, gen_sg)
    target_bd = lax.stop_gradient(gen_sg + drift)

    losses = jnp.sum((gen_bd - target_bd) ** 2, axis=-1)
    return jnp.mean(losses)


@nnx.jit
def train_step(model, optimizer, z_bd, datasample_bd):
    
    grad_fn = nnx.value_and_grad(drift_loss, has_aux=False)
    loss, grads = grad_fn(model, z_bd, datasample_bd)

    optimizer.update(model, grads)

    return loss

