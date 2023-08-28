from torch import nn
from torch.nn import functional as F
from utils import *
import math


class PositionEmbedding(nn.Module):
	def __init__(self, d_model, max_len=5000):
		super(PositionEmbedding, self).__init__()
		pe = torch.zeros(max_len, d_model).float()
		pe.require_grad = False

		position = torch.arange(max_len).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * - (math.log(10000) / d_model)).exp()
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		# 把参数注册到state_dict,随模型移动cpu_gpu，但不更新
		# https://blog.csdn.net/weixin_46197934/article/details/119518497
		self.register_buffer('pe', pe)

	def forward(self, x):
		"""
		:param x: batch_size, seq_len, df
		:return: 1， seq_len, d_model
		"""
		return self.pe[:, :x.size(1), :]


class DataEmbedding(nn.Module):
	def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
		super(DataEmbedding, self).__init__()
		self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
		self.position_embedding = PositionEmbedding(d_model=d_model)
		if embed_type != 'timeF':
			# fixed,正余弦位置编码
			# learned,embedding
			self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
		else:
			# timef,线性映射
			self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, x_mark):
		# autoformer相比,多了一个位置信息的编码
		x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
		return x


class Encoder(nn.Module):
	def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
		super(Encoder, self).__init__()
		self.attn_layers = nn.ModuleList(attn_layers)
		self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
		self.norm = norm_layer

	def forward(self, x, attn_mask=None):
		"""
		:param x:batch_size,seq_len,d_model
		:param attn_mask:
		:return:
		"""
		attns = []
		if self.conv_layers is not None:
			# attn_layers要比conv_layers多一个
			for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
				x, attn = attn_layer(x, attn_mask=attn_mask)
				x = conv_layer(x)
				attns.append(attn)
			x, attn = self.attn_layers[-1](x)
			attns.append(attn)
		else:
			for attn_layer in self.attn_layers:
				x, attn = attn_layer(x, attn_mask=attn_mask)
				attns.append(attn)

		if self.norm is not None:
			x = self.norm(x)

		return x, attns


class EncoderLayer(nn.Module):
	def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
		super(EncoderLayer, self).__init__()
		d_ff = d_ff or d_model * 4
		self.attention = attention
		self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == 'relu' else F.gelu

	def forward(self, x, attn_mask=None):
		"""
		:param x: batch_size,seq_len,d_model
		:return:
		"""
		# 1, self_attention
		new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
		# 2, residual
		x = x + self.dropout(new_x)
		y = x = self.norm1(x)
		# 3,feed_forward, redidual
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))
		return self.norm2(x+y), attn


class AttentionLayer(nn.Module):
	def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
		super(AttentionLayer, self).__init__()
		d_keys = d_keys or d_model // n_heads
		d_values = d_values or d_model // n_heads
		self.inner_attention = attention
		self.query_projection = nn.Linear(d_model, d_keys * n_heads)
		self.key_projection = nn.Linear(d_model, d_keys * n_heads)
		self.value_projection = nn.Linear(d_model, d_values * n_heads)
		self.out_projection = nn.Linear(d_values * n_heads, d_model)
		self.n_heads = n_heads

	def forward(self, queries, keys, values, attn_mask):
		"""
		:param queries: batch_size,seq_len,d_model
		:param keys: batch_size,seq_len,d_model
		:param values: batch_size,seq_len,d_model

		:return:
		"""
		B, L, _ = queries.shape
		_, S, _ = keys.shape
		H = self.n_heads

		# batch_size,seq_len,n_heads, d_keys
		queries = self.query_projection(queries).view(B, L, H, -1)
		keys = self.key_projection(keys).view(B, S, H, -1)
		values = self.value_projection(values).view(B, S, H, -1)

		out, attn = self.inner_attention(queries, keys, values, attn_mask)
		out = out.view(B, L, -1)
		return self.out_projection(out), attn


class ProbMask():
	def __init__(self, B, H, L, index, scores, device='cpu'):
		"""
		:param B: batch_size
		:param H: n_heads
		:param L: len_q
		:param index: batch_size, n_heads, u
		:param scores: batch_size, n_heads, u, len_kv
		:param device:
		"""
		# len_q, len_kv
		_mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
		# batch_size, n_heads, len_q, len_kv
		_mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
		# batch_size, 1, 1
		# 1, n_heads, 1
		# batch_size, n_heads, u
		# batch_size, n_heads, u, len_kv
		indicator = _mask_ex[
			torch.arange(B)[:, None, None],
			torch.arange(H)[None, :, None],
			index,
			:
		].to(device)
		self._mask = indicator.view(scores.shape).to(device)

	@property
	def mask(self):
		return self._mask


class ProbAttention(nn.Module):
	def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
		super(ProbAttention, self).__init__()
		self.factor = factor
		self.scale = scale
		self.mask_flag = mask_flag
		self.output_attention = output_attention
		self.attention_dropout = attention_dropout

	def _prob_QK(self, Q, K, sample_k, n_top):
		"""
		:param Q: batch_size, n_heads, len_q, d_keys
		:param K: batch_size, n_heads, len_kv, d_keys
		:param sample_k:
		:param n_top:
		:return:
		"""
		B, H, L_K, E = K.shape
		_, _, L_Q, _ = Q.shape

		# batch_size, n_heads, len_q, len_kv, d_keys
		K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
		# 每个q抽样k的索引, len_q, sample_k
		index_sample = torch.randint(L_K, (L_Q, sample_k))
		# 多维数组切片,要求数组形状相同(可扩展)
		# len_q, 1
		# len_q, sample_k
		# batch_size, n_heads, len_q, sample_k, d_keys
		K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
		# batch_size, n_heads, len_q, 1, d_keys
		# batch_size, n_heads, len_q, d_keys, sample_k
		# batch_size, n_heads, len_q, sample_k
		Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
		# 稀疏性定义，用来筛选可用q_i, max(scores) - mean(scores)
		# batch_size, n_heads, len_q
		M = Q_K_sample.max(dim=-1)[0] - torch.div(Q_K_sample.sum(dim=-1), L_K)
		# q_i的索引, batch_size, n_heads, n_top
		M_top = M.topk(n_top, dim=-1, sorted=False)[1]

		# batch_size, n_heads, n_top, d_keys
		Q_reduce = Q[
			torch.arange(B)[:, None, None],
			torch.arange(H)[None, :, None],
			M_top,
			:
		]
		# batch_size, n_heads, n_top, d_keys
		# batch_size, n_heads, d_keys, len_kv
		# batch_size, n_heads, n_top, len_kv
		Q_K = torch.matmul(Q_reduce, K.tranpose(-2, -1))

		return Q_K, M_top

	def _get_intial_context(self, V, L_Q):
		"""
		:param V: batch_size, n_heads, len_kv, d_keys
		:param L_Q: len_q
		:return:
		"""
		B, H, L_V, D = V.shape
		if not self.mask_flag:
			# 在encoder阶段self_attention，对于不计算q的用values在len_kv的均值填充
			V_sum = V.mean(dim=-2)
			# batch_size,n_heads,len_q, d_keys
			context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
		else:
			# 在decoder阶段self_attention, 用cumsum(cummean,masked不能直接mean)来填充
			assert (L_Q == L_V)
			context = V.cumsum(dim=-2)
		return context

	def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
		"""

		:param context_in: batch_size, n_heads, len_q, d_keys
		:param V: batch_size, n_heads, len_kv, d_keys
		:param scores: batch_size, n_heads, u, len_kv
		:param index: batch_size, n_heads, u
		:param L_Q: len_q
		:param attn_mask:
		:return:
		"""
		B, H, L_V, D = V.shape

		if self.mask_flag:
			# 在decoder阶段self_attention,进行maksed_fill
			# batch_size, n_heads, u, len_kv
			attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
			scores.masked_fill_(attn_mask, -np.inf)
		attn = torch.softmax(scores, dim=-1)

		context_in[
			torch.arange(B)[:, None, None],
			torch.arange(H)[None, :, None],
			index,
			:
		] = torch.matmul(attn, V).type_as(context_in)

		if self.output_attention:
			# 稀疏得分定义和1/len_kv的距离，均值填充来自于此
			# 在decoder self_attention阶段，attns应该是下三角
			attns = (torch.ones([B, H, L_Q, L_V]) / L_V).type_as(attn).to(attn.device)
			attns[
				torch.arange(B)[:, None, None],
				torch.arange(H)[None, :, None],
				index,
				:
			] = attn
			return context_in, attns
		else:
			return context_in, None

	def forward(self, queries, keys, values, attn_mask):
		"""
		:param queries: batch_size, len_q, n_heads, d_keys
		:param keys: batch_size, len_kv, n_heads, d_keys
		:param values: batch_size, len_kv, n_heads, d_keys
		:return:
		"""
		B, L_Q, H, D = queries.shape
		_, L_K, _, _ = keys.shape

		queries = queries.transpose(2, 1)
		keys = keys.transpose(2, 1)
		values = values.transpose(2, 1)

		# informer的重要改造就是降低attention的计算时间复杂度
		# 1， 从整体attention_score来说是一个稀疏矩阵，大部分之间是不相关的，往往高度相关的得分呈现长尾分布
		# 2， 对于q_i来说，如果有高度相关的得分，那么最大值得分和平均值得分有很大差距；如果没有，则没有差距，这个差距描述为稀疏性得分
		# 3， 对于q_i来说，随机抽样多个k_i, 计算稀疏性得分
		# 4,  根据稀疏性得分选出最大的几个q_i
		# 5, 其余的为计算得分的q_i用平均值代替
		# 详细步骤参考 Solve_Challenge_1 https://blog.csdn.net/weixin_43332715/article/details/124885959

		# 对每个q_i采样factor*log(l_k)个key计算稀疏性得分
		U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
		U_part = U_part if U_part < L_K else L_K
		# 按照稀疏性得分选择factor*log(l_q)个query
		u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
		u = u if u < L_Q else L_Q

		# u个q_i的att_score, batch_size, n_heads, u, len_kv
		# u个q_i的index, batch_size, n_heads, u
		scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
		scale = self.scale or 1.0 / math.sqrt(D)
		if scale is not None:
			scores_top = scores_top * scale

		# batch_size,n_heads,len_q, d_keys
		# 初始化atten输出，用序列方向的平均值填充
		context = self._get_intial_context(values, L_Q)
		# 根据u个q_i的att_score更新atten输出
		# batch_size,n_heads,len_q, d_keys
		# batch_size,n_heads,len_q, len_kv
		context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

		return context.contiguous(), attn


class ConvLayer(nn.Module):
	def __init__(self, c_in):
		"""
		1, 降低dec部分attention的计算量；
		2，浓缩特征，是特征更明显更有规律
		:param c_in:
		"""
		super(ConvLayer, self).__init__()
		self.downconv = nn.Conv1d(c_in, c_in, 3, padding=1, padding_mode='circular')
		self.norm = nn.BatchNorm1d(c_in)
		self.activation = nn.ELU()
		self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

	def forward(self, x):
		x = self.downconv(x.permute(0, 2, 1))
		x = self.norm(x)
		x = self.activation(x)
		x = self.maxpool(x)
		x = x.transpose(1, 2)
		return x


class DecoderLayer(nn.Module):
	def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
		super(DecoderLayer, self).__init__()
		d_ff = d_ff or 4 * d_model
		self.self_attention = self_attention
		self.cross_attention = cross_attention
		self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.norm3 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.relu if activation == 'relu' else F.gelu

	def forward(self, x, cross, x_mask=None, cross_mask=None):
		# 1, self_attention, residual
		x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
		# batch_size, label_len+pred_len, d_model
		x = self.norm1(x)
		# 2, cross_attenion, residual
		x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
		y = x = self.norm2(x)
		# 3, feed_forward, residual
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))
		return self.norm3(x + y)


class Decoder(nn.Module):
	def __init__(self, layers, norm_layer=None, projection=None):
		super(Decoder, self).__init__()
		self.layers = nn.ModuleList(layers)
		self.norm = norm_layer
		self.projection = projection

	def forward(self, x, cross, x_mask=None, cross_mask=None):
		"""
		:param x: batch_size, label_len+pred_len, d_model
		:param cross: batch_size, seq_len, d_model
		:param x_mask: None
		:param cross_mask: None
		:return:
		"""
		for layer in self.layers:
			x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

		if self.norm is not None:
			x = self.norm(x)

		if self.projection is not None:
			x = self.projection(x)

		return x


"""
informer细节参考， https://blog.csdn.net/weixin_43332715/article/details/124885959
重点：
	1， 利用抽样来计算稀疏得分来降低attention_score的计算复杂度
	2, 	不采用传统时间预测的step-to_step模型，一步到位，减少误差积累
	3， 还是传统transformer的结构， 其中decoder中self_attention 保留了masked
"""
class Informer(nn.Module):
	def __init__(self, configs):
		super(Informer, self).__init__()
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
						attention=ProbAttention(
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
			conv_layers=[
				ConvLayer(configs.d_model) for _ in range(configs.e_layers - 1)
			] if configs.distil else None,
			norm_layer=nn.LayerNorm(configs.d_model)
		)
		# decoder
		self.decoder = Decoder(
			layers=[
				DecoderLayer(
					self_attention=AttentionLayer(
						attention=ProbAttention(
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
						attention=ProbAttention(
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
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		"""
		x_enc, batch_size, seq_len, df
		x_mark_enc, batch_size, seq_len, dt
		x_dec, batch_size, label_len+pred_len, df
		x_mark_dec, batch_size, label_len+pred_len,dt
		"""
		# encoder
		# batch_size, seq_len, d_model
		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		# batch_size, seq_len, d_model
		# batch_size, n_heads, seq_len, seq_len
		enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

		# batch_size, label_len+pred_len, d_model
		dec_out = self.dec_embedding(x_dec, x_mark_dec)
		dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

		if self.output_attention:
			return dec_out[:, -self.pred_len, :], attns
		else:
			return dec_out[:, -self.pred_len, :]

