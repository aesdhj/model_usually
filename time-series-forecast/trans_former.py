from torch import nn
from in_former import *


class TriangularCausalMask():
	def __init__(self, B, L, device='cpu'):
		mask_shape = [B, 1, L, L]
		with torch.no_grad():
			# transformer,dec_mask
			self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

	@property
	def mask(self):
		return self._mask


class FullAttention(nn.Module):
	def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
		super(FullAttention, self).__init__()
		self.scale = scale
		self.mask_flag = mask_flag
		self.dropout = nn.Dropout(attention_dropout)
		self.output_attention = output_attention

	def forward(self, queries, keys, values, attn_mask):
		# batch_size, seq_len_q, h_heads, d_keys
		B, L, H, E = queries.shape
		# batch_size, seq_len_kv, h_heads, d_keys
		B, S, H, D = values.shape
		scale = self.scale or 1.0 / math.sqrt(E)

		# batch_size, h_heads, seq_len_q, seq_len_kv
		# https://zhuanlan.zhihu.com/p/361209187, ein求和，注意自由索引和求和索引
		scores = torch.einsum('blhe,bshe->bhls', queries, keys)
		if self.mask_flag:
			if attn_mask is None:
				# batch_size, 1, seq_len_q, seq_len_q
				attn_mask = TriangularCausalMask(B, L, device=queries.device)
			# dec_self att截断本省只能跟本省和序列之前的相关
			scores.masked_fill_(attn_mask, -np.inf)

		A = self.dropout(torch.softmax(scores, dim=-1))
		V = torch.einsum('bhls,bshd->blhd', A, values)

		if self.output_attention:
			return (V.contiguous(), A)
		else:
			return (V.contiguous(), None)


class Transformer(nn.Module):
	def __init__(self, configs):
		super(Transformer, self).__init__()
		self.pred_len = configs.pred_len
		self.output_attention = configs.output_attention

		# embedding
		self.enc_embedding = DataEmbedding(
			configs.enc_in, configs.d_model, configs.embed,
			configs.freq, configs.dropout
		)
		self.dec_embedding = DataEmbedding(
			configs.dec_in, configs.d_model, configs.embed,
			configs.freq, configs.dropout
		)
		# encoder
		self.encoder = Encoder(
			attn_layers=[
				EncoderLayer(
					attention=AttentionLayer(
						attention=FullAttention(
							mask_flag=False,
							factor=configs.factor,
							scale=None,
							attention_dropout=configs.dropout,
							output_attention=configs.output_attention
						),
						d_model=configs.d_model,
						n_heads=configs.n_heads
					),
					d_model=configs.d_model,
					d_ff=configs.d_ff,
					dropout=configs.dropout,
					activation=configs.activation
				) for _ in range(configs.e_layers)
			],
			conv_layers=None,
			norm_layer=nn.LayerNorm(configs.d_model)
		)
		# decoder
		self.decoder = Decoder(
			layers=[
				DecoderLayer(
					self_attention=AttentionLayer(
						attention=FullAttention(
							mask_flag=True,
							factor=configs.factor,
							scale=None,
							attention_dropout=configs.dropout,
							output_attention=configs.output_attention
						),
						d_model=configs.d_model,
						n_heads=configs.n_heads
					),
					cross_attention=AttentionLayer(
						attention=FullAttention(
							mask_flag=False,
							factor=configs.factor,
							scale=None,
							attention_dropout=configs.dropout,
							output_attention=configs.output_attention
						),
						d_model=configs.d_model,
						n_heads=configs.n_heads
					),
					d_model=configs.d_model,
					d_ff=configs.d_ff,
					dropout=configs.dropout,
					activation=configs.activation
				) for _ in range(configs.d_layers)
			],
			norm_layer=nn.LayerNorm(configs.d_model),
			projection=nn.Linear(configs.d_model, configs.c_out)
		)

	def forward(
			self, x_enc, x_mark_enc, x_dec, x_mark_dec,
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
	):
		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

		dec_out = self.dec_embedding(x_dec, x_mark_dec)
		dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

		if self.output_attention:
			return dec_out[:, -self.pred_len:, :], attns
		else:
			return dec_out[:, -self.pred_len:, :]

