from torch import nn
from utils import *


"""
dliner, nliner细节参考， https://zhuanlan.zhihu.com/p/521329803
重点：
	1， 利用简单的线性变换分别预测趋势和周期
	2, 	作者提出传统transformer的self_attention是无序操作，一定要加上pos_embedding才有效果
	3， 不采用传统时间预测的step-to_step模型，一步到位，减少误差积累
彩蛋: transformer的改造是否真的对时序预测有意义， 细节参考 https://www.zhihu.com/question/493821601
	autoformer作者回答 https://www.zhihu.com/question/493821601/answer/2349331444
		总结:
			1, autoformer对长序列有优势
	dliner作者回答 https://www.zhihu.com/question/493821601/answer/2506641761
		总结： 
			1， transformer 中的self_attention对时序不敏感的的无效操作
			2， dliner作为时序的pre_study,应该作为时序模型的基准模型
"""
class DLinear(nn.Module):
	def __init__(self, configs):
		super(DLinear, self).__init__()
		self.individual = configs.individual
		self.seq_len = configs.seq_len
		self.pred_len = configs.pred_len
		self.channels = configs.enc_in
		self.output_attention = configs.output_attention
		# decomp
		kernel_size = configs.moving_avg
		self.decompsition = series_decomp(kernel_size)

		if self.individual:
			self.linear_seasonal = nn.ModuleList()
			self.linear_trend = nn.ModuleList()
			for i in range(self.channels):
				self.linear_seasonal.append(nn.Linear(self.seq_len, self.pred_len))
				self.linear_trend.append(nn.Linear(self.seq_len, self.pred_len))
		else:
			self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
			self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

			# self.linear_seasonal = nn.Linear(self.seq_len * self.channels, self.pred_len * self.channels)
			# self.linear_trend = nn.Linear(self.seq_len * self.channels, self.pred_len * self.channels)

	def forward(
			self, x_enc, x_mark_enc, x_dec, x_mark_dec,
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		# batch_size, seq_len, d_f
		seasonal_init, trend_init = self.decompsition(x_enc)
		# batch_size, d_f, seq_len
		seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

		# 每个特征是不是单独线性变化
		if self.individual:
			# batch_size, d_f, pred_len
			seasonal_output = torch.zeros(
				[seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
				dtype=seasonal_init.dtype
			).to(seasonal_init.device)
			trend_output = torch.zeros(
				[trend_init.size(0), trend_init.size(1), self.pred_len],
				dtype=trend_init.dtype
			).to(trend_init.device)
			for i in range(self.channels):
				seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
				trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
		else:
			seasonal_output = self.linear_seasonal(seasonal_init)
			trend_output = self.linear_trend(trend_init)

			# seasonal_output = self.linear_seasonal(seasonal_init.view(seasonal_init.size(0), -1))
			# trend_output = self.linear_trend(trend_init.view(trend_init.size(0), -1))
			# seasonal_output = seasonal_output.view(seasonal_init.size(0), self.channels, -1)
			# trend_output = trend_output.view(trend_init.size(0), self.channels, -1)

		output = seasonal_output + trend_output
		if self.output_attention:
			return output.permute(0, 2, 1), None
		else:
			return output.permute(0, 2, 1)


class NLinear(nn.Module):
	def __init__(self, configs):
		super(NLinear, self).__init__()
		self.individual = configs.individual
		self.seq_len = configs.seq_len
		self.pred_len = configs.pred_len
		self.channels = configs.enc_in
		self.output_attention = configs.output_attention

		if self.individual:
			self.linear = nn.ModuleList()
			for i in range(self.channels):
				self.linear.append(nn.Linear(self.seq_len, self.pred_len))
		else:
			self.linear = nn.Linear(self.seq_len, self.pred_len)

	def forward(
			self, x_enc, x_mark_enc, x_dec, x_mark_dec,
			enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
		seq_last = x_enc[:, -1:, :].detach()
		# batch_size, seq_len, d_f
		# 可以理解为以最后一个输入值为基准的浮动预测
		x_enc = x_enc - seq_last
		# 每个特征是不是单独线性变化
		if self.individual:
			# batch_size, pred_len, d_f
			output = torch.zeros(
				[x_enc.size(0), self.pred_len, x_enc.size(2)],
				dtype=x_enc.dtype
			).to(x_enc.device)
			for i in range(self.channels):
				output[:, :, i] = self.linear[i](x_enc[:, :, i])
		else:
			output = self.linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

		output = output + seq_last
		if self.output_attention:
			return output, None
		else:
			return output