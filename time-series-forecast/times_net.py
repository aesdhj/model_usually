from utils import *
from torch import nn
from torch.nn import functional as F
from in_former import DataEmbedding


def FFT_for_Period(x, k):
	"""
	:param x: batch_size, seq_len+pred_len, d_model
	:param k:
	:return:
	"""
	# batch_size, (seq_len+pred_len)//2+1, d_model
	xf = torch.fft.rfft(x, dim=1)
	# 频域是多个振幅不同的波信息，振幅越大对原时序影响越大，频率越大波在原时序重复的次数越多
	# (seq_len+pred_len)//2+1
	frequency_list = abs(xf).mean(dim=0).mean(dim=-1)
	frequency_list[0] = 0
	# 索引就是频率的大小
	_, top_list = torch.topk(frequency_list, k)
	# 周期 = 时序长度/频率(时序长度内)
	period = x.shape[1] // top_list
	return period, abs(xf).mean(dim=-1)[:, top_list]


class Inception_Block_V1(nn.Module):
	def __init__(self, in_channels, out_channels, num_kernels=6, init_weights=True):
		super(Inception_Block_V1, self).__init__()
		kernels = []
		for i in range(num_kernels):
			kernels.append(
				nn.Conv2d(in_channels, out_channels, kernel_size=2*i+1, padding=i)
			)
		self.kernels = nn.ModuleList(kernels)
		if init_weights:
			self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# fan_in正向传播方差不变，fan_out反向传播反差不变
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		res_list = []
		for i in range(self.num_kernels):
			res_list.append(self.kernels[i](x))
		res = torch.stack(res_list, dim=-1).mean(dim=-1)
		return res


class TimesBlock(nn.Module):
	def __init__(self, configs):
		super(TimesBlock, self).__init__()
		self.k = configs.top_k

		self.conv = nn.Sequential(
			Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
			nn.GELU(),
			Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
		)

	def forward(self, x):
		"""
		:param x: batch_size, seq_len+pred_len, d_model
		:return:
		"""
		B, T, N = x.size()
		# k, (batch_size, k)
		period_list, period_weight = FFT_for_Period(x, self.k)

		res = []
		for i in range(self.k):
			period = period_list[i]
			# 填充长度
			if (self.seq_len + self.pred_len) % period != 0:
				length = ((self.seq_len + self.pred_len) // period + 1) * period
				padding = torch.zeros([x.shape[0], length-(self.seq_len + self.pred_len), x.shape[2]]).to(x.device)
				out = torch.cat([x, padding], dim=1)
			else:
				length = (self.seq_len + self.pred_len)
				out = x

			# reshape的目的是将以为数据转化为二维数据，原时序数据转化为多个周期数据并列
			# reshape后的数据类似cv数据， 用conv2d来提取周期之间和周期内的数据
			# batch_size, d_model, (seq_len+pred_len)//period, period
			out = out.reshape(B, length//period, period, N)
			# batch_size, d_model, (seq_len+pred_len)//period, period
			out = self.conv(out)
			# batch_size, seq_len+pred_len, d_model
			out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
			res.append(out[:, :(self.seq_len + self.pred_len), :])
		# batch_size, seq_len+pred_len, d_model, k
		res = torch.stack(res, dim=-1)
		# 用period的振幅作为权重
		period_weight = F.softmax(period_weight, dim=1)
		# batch_size, seq_len+pred_len, d_model, k
		period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
		# batch_size, seq_len+pred_len, d_model
		res = torch.sum(res * period_weight, dim=-1)
		res = res + x
		return res


"""
TimesNet细节参考， https://zhuanlan.zhihu.com/p/606575441
重点： 
	1， 用频域振幅提取相对应的频率，把原时序切分
	2， 将切分的数据用conv2d提取周期内和周期间的数据
"""
class TimesNet(nn.Module):
	def __init__(self, configs):
		super(TimesNet, self).__init__()
		self.configs = configs
		self.seq_len = configs.seq_len
		self.pred_len = configs.pred_len
		self.e_layers = configs.e_layers

		self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
		self.predict_linear = nn.Linear(self.seq_len, self.seq_len+self.pred_len)
		self.layer_norm = nn.LayerNorm(configs.d_model)
		self.model = nn.ModuleList([
			TimesBlock(configs) for _ in range(self.e_layers)
		])
		self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

	def forward(
			self, x_enc, x_mark_enc, x_dec, x_mark_dec,
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		"""
		x_enc, batch_size, seq_len, df
		x_mark_enc, batch_size, seq_len, dt
		"""
		# Normalization from Non-stationary Transformer
		# batch_size, seq_len, df
		means = x_enc.mean(dim=1, keepdim=True).detach()
		x_enc = x_enc - means
		stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
		x_enc = x_enc / stdev

		# embedding
		# batch_size, seq_len, d_model
		enc_out = self.enc_embedding(x_enc, x_mark_enc)
		# batch_size, seq_len+pred_len, d_model
		enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

		# TimesNet
		for i in range(self.e_layers):
			# batch_size, seq_len+pred_len, d_model
			enc_out = self.layer_norm(self.model[i](enc_out))
		# batch_size, seq_len+pred_len, df
		dec_out = self.projection(enc_out)

		# De-Normalization from Non-stationary Transformer
		dec_out = dec_out * (
			stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len+self.pred_len, 1)
		)
		dec_out = dec_out + (
			means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len+self.pred_len, 1)
		)
		if self.output_attention:
			return dec_out[:, -self.pred_len:, :], None
		else:
			return dec_out[:, -self.pred_len:, :]

