from utils import *
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def squeeze_last_dim(x):
	if len(x.shape) == 3 and x.shape[-1] == 1:
		return x[:, :, 0]
	return x


class Block(nn.Module):
	def __init__(self, units, thetas_dim, backcast_length, forecast_length, share_thetas=False, nb_harmonics=None):
		super(Block, self).__init__()

		self.fc1 = nn.Linear(backcast_length, units)
		self.fc2 = nn.Linear(units, units)
		self.fc3 = nn.Linear(units, units)
		self.fc4 = nn.Linear(units, units)
		if share_thetas:
			self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
		else:
			self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)
			self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
		self.backcast_linespace = np.arange(backcast_length) / backcast_length
		self.forecast_linespcae = np.arange(forecast_length) / forecast_length

	def forward(self, x):
		x = squeeze_last_dim(x)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		return x


class TrendBlock(Block):
	def __init__(self, units, thetas_dim, backcast_length, forecast_length, share_thetas=True, nb_harmonics=None):
		super(TrendBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length, share_thetas, nb_harmonics)

	def trend_model(self, thetas, t):
		"""
		:param thetas: (batch_size, thetas_dim)
		:param t: (backcast_length,)/(forecast_length,)
		:return:
		"""
		p = thetas.size()[-1]
		assert p <= 4, 'thetas_dim is too big.'
		# thetas_dim, backcast_length/forecast_length
		T = torch.tensor(np.array([t ** i for i in range(p)])).float()
		# batch_size, backcast_length/forecast_length
		return thetas.mm(T.to(thetas.device))

	def forward(self, x):
		"""
		用相对时间轴归一化后的多阶多项式线性拟合趋势
		:param x: batch_size, backcast_length
		:return:
		"""
		# batch_size, units
		x = super(TrendBlock, self).forward(x)
		# batch_size, backcast_length
		backcast = self.trend_model(self.theta_b_fc(x), self.backcast_linespace)
		# batch_size, forecast_length
		forecast = self.trend_model(self.theta_f_fc(x), self.forecast_linespcae)
		return backcast, forecast


class SeasonalityBlock(Block):
	def __init__(self, units, thetas_dim, backcast_length, forecast_length, share_thetas=True, nb_harmonics=None):
		if nb_harmonics:
			super(SeasonalityBlock, self).__init__(units, nb_harmonics, backcast_length, forecast_length, share_thetas, nb_harmonics)
		else:
			super(SeasonalityBlock, self).__init__(units, forecast_length, backcast_length, forecast_length, share_thetas, nb_harmonics)

	def seasonality_model(self, thetas, t):
		"""
		:param thetas: (batch_size, thetas_dim)
		:param t: (backcast_length,)/(forecast_length,)
		:return
		"""
		p = thetas.size()[-1]
		assert p <= thetas.shape[1], 'thetas_dim is too big.'
		p1, p2 = (p//2, p//2) if p % 2 == 0 else (p//2, p//2+1)
		# 不同周期的正弦余弦函数，周期分别为backcast_length/i, forecast_length/i
		s1 = torch.tensor(np.array([np.cos(2*np.pi*i*t) for i in range(p1)])).float()
		s2 = torch.tensor(np.array([np.sin(2*np.pi*i*t) for i in range(p2)])).float()
		# thetas_dim, backcast_length/forecast_length
		S = torch.cat([s1, s2])
		# batch_size, backcast_length/forecast_length
		return thetas.mm(S.to(thetas.device))

	def forward(self, x):
		"""
		傅里叶级数,用不同周期的正弦余弦函数去拟合周期性
		:param x: batch_size, backcast_length
		:return:
		"""
		# batch_size, units
		x = super(SeasonalityBlock, self).forward(x)
		# batch_size, backcast_length
		backcast = self.seasonality_model(self.theta_b_fc(x), self.backcast_linespace)
		# batch_size, forecast_length
		forecast = self.seasonality_model(self.theta_f_fc, self.forecast_linespcae)
		return backcast, forecast


class GenericBlock(Block):
	def __init__(self, units, thetas_dim, backcast_length, forecast_length, share_thetas=True, nb_harmonics=None):
		super(GenericBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length, share_thetas, nb_harmonics)
		self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
		self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

	def forward(self, x):
		"""
		线性去拟合残差
		:param x: batch_size, backcast_length
		:return:
		"""
		# batch_size, units
		x = super(GenericBlock, self).forward(x)

		# batch_size, thetas_dim
		theta_b = self.theta_b_fc(x)
		theta_f = self.theta_f_fc(x)

		# batch_size, backcast_length
		backcast = self.backcast_fc(theta_b)
		# batch_size, forecast_length
		forecast = self.forecast_fc(theta_f)

		return backcast, forecast


"""
前置知识：
	离散傅里叶变换实际计算， https://www.zhihu.com/question/21314374/answer/542909849
NBeats细节详解，https://zhuanlan.zhihu.com/p/382516756
重点：
	1， 用高阶多项式为基础拟合趋势
	2， 用不同周期的正余弦为基础拟合周期性
	3， 类似GBDT的用上一步的残差作为下一个block的输入
"""
class NBeats(nn.Module):
	def __init__(self, configs):
		super(NBeats, self).__init__()
		self.forecast_length = configs.pred_len
		self.backcast_length = configs.seq_len
		self.stack_types = configs.stack_types
		self.nb_blocks_per_stack = configs.nb_blocks_per_stack
		self.share_weights_in_stack = configs.share_weights_in_stack
		self.hidden_layer_units = configs.hidden_layer_units
		self.thetas_dim = configs.thetas_dim
		self.nb_harmonics = configs.nb_harmonics
		self.share_thetas = configs.share_thetas

		self.stacks = []
		for stack_id in range(len(self.stack_type)):
			self.stacks.append(self.create_stack(stack_id))

	def create_stack(self, stack_id):
		stack_type = self.stack_types[stack_id]
		blocks = []
		for block_id in range(self.nb_blocks_per_stack):
			block_init = NBeats.select_block(stack_type)
			# 在同一个stack 共享权重
			if self.share_weights_in_stack and block_id != 0:
				block = blocks[-1]
			else:
				block = block_init(
					self.hidden_layer_units, self.thetas_dim[stack_id],
					self.backcast_length, self.forecast_length,
					share_thetas=self.share_thetas,
					nb_harmonics=self.nb_harmonics
				)
				blocks.append(block)
		return blocks

	@staticmethod
	def select_block(self, block_type):
		if block_type == 'trend':
			return TrendBlock
		elif block_type == 'seasonality':
			return SeasonalityBlock
		else:
			return GenericBlock

	def forward(
			self, x_enc, x_mark_enc, x_dec, x_mark_dec,
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None
	):
		"""
		https://github.com/philipperemy/n-beats
		x_enc, batch_size, seq_len, df
		"""
		assert x_enc.shape[-1] == 1, 'NBeats only support single feature'
		# batch_size, backcast_length
		backcast = squeeze_last_dim(x_enc)
		# batch_size, forecast_length
		forecast = torch.zeros(size=(backcast.shape[0], self.forecast_length))
		for stack_id in range(len(self.stacks)):
			for block_id in range(len(self.stacks[stack_id])):
				# batch_size, backcast_length
				# batch_size, forecast_length
				b, f = self.stacks[stack_id][block_id](backcast)
				# backcast残差作为bock的输入
				backcast = backcast - b
				# forecast累加
				forecast = forecast + f
		if self.output_attention:
			return forecast, None
		else:
			return forecast

