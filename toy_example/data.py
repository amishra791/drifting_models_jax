import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx

def sample_checkerboard(n: int, noise: float, rngs: nnx.Rngs) -> jax.Array: 
    b = jr.randint(rngs(), (n,), 0, 2)
    i = jr.randint(rngs(), (n,), 0, 2) * 2 + b
    j = jr.randint(rngs(), (n,), 0, 2) * 2 + b
    u = jr.uniform(rngs(), (n,))
    v = jr.uniform(rngs(), (n,))

    pts = jnp.stack([i + u, j + v]) + jr.normal(rngs(), (2, n)) * noise

    return pts

def sample_swiss_roll(n: int, noise: float, rngs: nnx.Rngs) -> jax.Array:
    u = jr.uniform(rngs(), (n,))
    t = 0.5 * jnp.pi + 4.0 * jnp.pi * u
    pts = jnp.stack([t * jnp.cos(t), t * jnp.sin(t)])
    pts = pts / (jnp.max(jnp.abs(pts)) + 1e-8)
    pts = pts + noise * jr.normal(rngs(), pts.shape)
    return pts