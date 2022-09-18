from turtle import forward
import torch
import torch.nn as nn
import torch.functional as F
import math

class RoFormerSelfAttention(nn.Module):
    # https://github.com/huggingface/transformers/blob/2c8b508ccabea6638aa463a137852ff3b64be036/src/transformers/models/roformer/modeling_roformer.py#L209
    def __init__(
        self, num_attention_heads: int, hidden_size: int,
        rotary_value: bool = False, attention_probs_dropout_prob: float = 0,
        is_decoder: bool = True
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        self.is_decoder = is_decoder
        self.rotary_value = rotary_value

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            if sinusoidal_pos is not None:
                if self.rotary_value:
                    query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer, value_layer
                    )
                else:
                    query_layer, key_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer
                    )
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RoFormerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


class PITransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, num_heads: int = 4) -> None:
        super().__init__()
        self.alpha = nn.Parameter(data=torch.randn(1))
        self.num_heads =num_heads
        self.self_attn = RoFormerSelfAttention(
            hidden_size=d_model,
            num_attention_heads=num_heads,
        )
        self.ff = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.ReLU(),
            nn.Linear(in_features=d_ff, out_features=d_model)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        mask = mask_gen(x, self.num_heads)
        x = x + self.alpha * self.self_attn(hidden_states=x, attention_mask=mask)[0]
        x = x + self.alpha * self.ff(x)
        return x
    
class LogMeanScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", None)
        
    def forward(self, x: torch.Tensor, shift: int, window_size: int, inverse: bool = False) -> torch.Tensor:
        if shift == 0:
            mean_slice = slice(-window_size, None)
        else:
            mean_slice = slice(-window_size - shift, -shift)

        if inverse:
            return torch.exp(x) * self.mu.repeat(1, x.shape[1]).unsqueeze(-1)
        else:
            self.mu = torch.mean(x[:, mean_slice, :], dim=1)
            return torch.log(x / self.mu.repeat(1, x.shape[1]).unsqueeze(-1))


class PITransformer(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, num_heads: int =  4, n_layers: int = 4) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.scaler = LogMeanScaler()
        self.transformer = nn.Sequential(
            nn.Linear(in_features=1, out_features=d_model),
            *[PITransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads) for _ in range(n_layers)],
            nn.Linear(in_features=d_model, out_features=1)
        )
        
    def forward(self, x: torch.Tensor, shift: int, window_size: int, mask=None) -> torch.Tensor:
        x = self.scaler(x, inverse=False, shift=shift, window_size=window_size)
        x = self.transformer(x)
        x = self.scaler(x, inverse=True, shift=shift, window_size=window_size)
        return x

def mask_gen(x, n_heads):
    mask = torch.tril(
        torch.ones((x.shape[1], x.shape[1])),
        diagonal=0
    )
    
    mask[mask == 0] = float('-inf')
    mask[mask == 1] = 0.0
    
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], n_heads, 1, 1)
    return mask


if __name__ == "__main__":
    x = torch.randn(2, 10, 512)
    n_heads = 4
    mask = torch.tril(
        torch.ones((10, 10)),
        diagonal=0
    )
    
    mask[mask == 0] = float('-inf')
    mask[mask == 1] = 0.0
    
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], n_heads, 1, 1)
    
    print(mask)
    print(mask.shape)
    model = PITransformerBlock(num_heads=n_heads)
    y = model(x, mask)
    print(y)
    print(y.shape)
    
    
    x = torch.randn(2, 10, 1) + 100
    
    model = PITransformer(horizon=10)
    
    y = model(x, shift=2, window_size=2)
    
    print(y)