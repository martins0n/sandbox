import torch
import torch.nn as nn

torch.manual_seed(0)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_weight = nn.parameter.Parameter(torch.rand((embed_dim, embed_dim)))
        self.key_weight = nn.parameter.Parameter(torch.rand((embed_dim, embed_dim)))
        self.value_weight = nn.parameter.Parameter(torch.rand((embed_dim, embed_dim)))

    def forward(self, query, key, value):
        """
        query: [batch_size, q_seq_len, embed_dim]
        key: [batch_size, k_seq_len, embed_dim]
        value: [batch_size, k_seq_len, embed_dim]
        """
        query_proj = torch.einsum(
            "bik,kj->bij", query, self.query_weight.T
        )  # [batch_size, q_seq_len, embed_dim]
        key_proj = torch.einsum(
            "bik,kj->bij", key, self.key_weight.T
        )  # [batch_size, k_seq_len, embed_dim]
        value_proj = torch.einsum(
            "bik,kj->bij", value, self.value_weight.T
        )  # [batch_size, k_seq_len, embed_dim]
        query_key = torch.einsum(
            "bik,bjk->bij", query_proj, key_proj
        )  # [batch_size, q_seq_len, k_seq_len]
        query_key = torch.softmax(
            query_key / self.embed_dim ** (0.5), dim=-1
        )  # [batch_size, q_seq_len, k_seq_len]
        attention = torch.einsum(
            "bij,bjk->bik", query_key, value_proj
        )  # [batch_size, q_seq_len, embed_dim]
        return attention, query_key


if __name__ == "__main__":
    EMB_DIM = 3
    attention_head = AttentionHead(EMB_DIM)
    query = torch.Tensor([[[1, 1, 2], [2, 1, 3]]])
    key = torch.Tensor([[[1, 1, 2], [2, 2, 2]]])
    value = torch.Tensor([[[4, 1, 2], [4, 3, 2]]])

    custom_output, custom_attn_weight = attention_head(query, key, value)

    torch_attention_head = nn.MultiheadAttention(
        EMB_DIM, 1, bias=False, batch_first=True
    )
    with torch.no_grad():
        torch_attention_head.in_proj_weight.copy_(
            nn.parameter.Parameter(
                torch.cat(
                    (
                        attention_head.query_weight.data.detach(),
                        attention_head.key_weight.data.detach(),
                        attention_head.value_weight.data.detach(),
                    )
                )
            )
        )
        torch_attention_head.out_proj.weight.copy_(
            nn.parameter.Parameter(torch.eye(EMB_DIM))
        )
    torch_output, torch_attn_weight = torch_attention_head(query, key, value)
    assert (custom_output - torch_output).abs().max().item() < 1e-6
    assert (custom_attn_weight - torch_attn_weight).abs().max().item() < 1e-6