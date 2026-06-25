from dataclasses import dataclass


@dataclass
class FlexTabConfig:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    bottleneck_size: int = 768
    layer_norm_eps: float = 1e-5
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
