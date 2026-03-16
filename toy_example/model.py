from flax import nnx

class ToyModel(nnx.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, rngs: nnx.Rngs):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layers = nnx.List(
            [
                nnx.Linear(in_dim, hidden_dim, rngs=rngs), nnx.silu, 
                nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), nnx.silu,
                nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), nnx.silu,
                nnx.Linear(hidden_dim, out_dim, rngs=rngs)
            ])
    
    def __call__(self, z_bd):
        out_bd = z_bd
        for layer in self.layers:
            out_bd = layer(out_bd)
        return out_bd