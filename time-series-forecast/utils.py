import os
import torch
from torch import nn
import time
import numpy as np
from auto_former import Autoformer
from in_former import Informer
from trans_former import Transformer
from re_former import Reformer
from d_n_linear import DLinear, NLinear
from fed_former import FEDformer
from times_net import TimesNet
from nbeats import NBeats
from patchtst import PatchTST
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import List
import matplotlib.pyplot as plt
import math


class TimeFeature:
	def __init__(self):
		pass

	# 可调用类，类似函数
	# def __call__(self, index(参数): pd.DatetimeIndex(类型)) -> np.ndarray(生成结果类型):
	def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
		pass

	# 直接打印类内部信息
	def __repr__(self):
		return self.__class__.__name__ + '()'


class SecondOfMinute(TimeFeature):
	def __call__(self, index):
		return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
	def __call__(self, index):
		return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
	def __call__(self, index):
		return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
	def __call__(self, index):
		return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
	def __call__(self, index):
		return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
	def __call__(self, index):
		return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
	def __call__(self, index):
		return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
	def __call__(self, index):
		# .isocalendar()当前日期的年份、第几周、周几(其中返回为元组)
		return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
	"""
	根据时间轴不同颗粒度提取时间特征
	"""
	features_by_offsets = {
		offsets.YearEnd: [],
		offsets.QuarterEnd: [MonthOfYear],
		offsets.MonthEnd: [MonthOfYear],
		offsets.Week: [DayOfMonth, WeekOfYear],
		offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
		offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
		offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
		offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
		offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
	}

	offset = to_offset(freq_str)
	for offset_type, feature_classes in features_by_offsets.items():
		if isinstance(offset, offset_type):
			return [cls() for cls in feature_classes]

	supported_freq_msg = f"""
	Unsupported frequency {freq_str}
	The following frequencies are supported:       
		Y   - yearly
			alias: A
		M   - monthly
		W   - weekly
		D   - daily
		B   - business days
		H   - hourly
		T   - minutely
			alias: min
		S   - secondly
		"""
	raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
	return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


class Dataset_ETT_minute(Dataset):
	def __init__(
			self, root_path, flag='train', size=None,
			features='S', data_path='ETTm1.csv',
			target='OT', scale=True, timeenc=0, freq='t'):
		# size [seq_len, label, pred_len]
		if size is None:
			self.seq_len = 24 * 4 * 4
			self.label_len = 24 * 4
			self.pred_len = 24 * 4
		else:
			self.seq_len = size[0]
			self.label_len = size[1]
			self.pred_len = size[2]

		assert flag in ['train', 'test', 'val']
		type_map = {'train': 0, 'val': 1, 'test': 2}
		self.set_type = type_map[flag]

		self.features = features
		self.target = target
		self.scale = scale
		self.timeenc = timeenc
		self.freq = freq

		self.root_path = root_path
		self.data_path = data_path

		self.__read_data__()

	def __read_data__(self):
		self.scaler = StandardScaler()
		df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

		# pred_len的边界, 所以- self.seq_len
		border1s = [
			0,
			12 * 30 * 24 * 4 - self.seq_len,
			12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
		]
		border2s = [
			12 * 30 * 24 * 4,
			12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
			12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
		]
		border1 = border1s[self.set_type]
		border2 = border2s[self.set_type]

		# 'M' 多序列预测多序列, 'MS' 多序列预测单序列, 'S' 单序列预测单序列
		if self.features == 'M' or self.features == 'MS':
			cols_data = df_raw.columns[1:]
			df_data = df_raw[cols_data]
		elif self.features == 'S':
			df_data = df_raw[[self.target]]

		# 预测前序列是否标准化
		if self.scale:
			train_data = df_data[border1s[0]:border2s[0]]
			self.scaler.fit(train_data.values)
			data = self.scaler.transform(df_data.values)
		else:
			data = df_data.values

		# 提取时间特征
		df_stamp = df_raw[['date']]
		df_stamp['date'] = pd.to_datetime(df_stamp['date'])
		if self.timeenc == 0:
			df_stamp['minute'] = df_stamp.apply(lambda x: x['date'].minute, axis=1)
			df_stamp['minute'] = df_stamp['minute'] // 15
			df_stamp['hour'] = df_stamp.apply(lambda x: x['date'].hour, axis=1)
			df_stamp['day'] = df_stamp.apply(lambda x: x['date'].day, axis=1)
			df_stamp['weekday'] = df_stamp.apply(lambda x: x['date'].weekday(), axis=1)
			df_stamp['month'] = df_stamp.apply(lambda x: x['date'].month, axis=1)
			data_stamp = df_stamp.drop('date', axis=1).values
		elif self.timeenc == 1:
			data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
			data_stamp = data_stamp.transpose(1, 0)

		self.data_x = data[border1:border2]
		self.data_y = data[border1:border2]
		self.data_stamp = data_stamp[border1:border2]

	def __getitem__(self, index):
		# (index, index+seq_len)
		s_begin = index
		s_end = s_begin + self.seq_len
		# (index+seq_len-label_len, index+seq_len+pred_len)
		r_begin = s_end - self.label_len
		r_end = r_begin + self.label_len + self.pred_len

		seq_x = self.data_x[s_begin:s_end]
		seq_y = self.data_y[r_begin:r_end]
		seq_x_mark = self.data_stamp[s_begin:s_end]
		seq_y_mark = self.data_stamp[r_begin:r_end]

		return seq_x, seq_y, seq_x_mark, seq_y_mark

	def __len__(self):
		return len(self.data_x) - self.seq_len - self.pred_len + 1

	def inverse_transform(self, data):
		return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
	def __init__(
			self, root_path, flag='pred', size=None,
			features='S', data_path='ETTm1.csv',
			target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
		# size [seq_len, label, pred_len]
		if size is None:
			self.seq_len = 24 * 4 * 4
			self.label_len = 24 * 4
			self.pred_len = 24 * 4
		else:
			self.seq_len = size[0]
			self.label_len = size[1]
			self.pred_len = size[2]

		assert flag in ['pred']

		self.features = features
		self.target = target
		self.scale = scale
		self.inverse = inverse
		self.timeenc = timeenc
		self.freq = freq
		self.cols = cols

		self.root_path = root_path
		self.data_path = data_path

		self.__read_data__()

	def __read_data__(self):
		self.scaler = StandardScaler()
		df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

		if self.cols is not None:
			cols = self.cols.copy()
			cols.remove(self.target)
		else:
			cols = list(df_raw.columns)
			cols.remove(self.target)
			cols.remove('date')
		df_raw = df_raw[['date'] + cols + [self.target]]

		border1 = len(df_raw) - self.seq_len
		border2 = len(df_raw)

		if self.features == 'M' or self.features == 'MS':
			cols_data = df_raw.columns[1:]
			df_data = df_raw[cols_data]
		elif self.features == 'S':
			df_data = df_raw[[self.target]]

		if self.scale:
			self.scaler.fit(df_data.values)
			data = self.scaler.transform(df_data.values)
		else:
			data = df_data.values

		tmp_stamp = df_raw[['date']][border1:border2]
		tmp_stamp['date'] = pd.todatetime(tmp_stamp['date'])
		# 从最后一个日期值延申出去pred_len
		if self.data_path == 'ETTm1.csv':
			pred_dates = pd.date_range(tmp_stamp['date'].values[-1], periods=self.pred_len + 1, freq='15min')
		else:
			pred_dates = pd.date_range(tmp_stamp['date'].values[-1], periods=self.pred_len + 1, freq=self.freq)
		df_stamp = pd.DataFrame(
			list(tmp_stamp['date'].values) + list(pred_dates[1:]),
			columns=['date']
		)
		if self.timeenc == 0:
			df_stamp['minute'] = df_stamp.apply(lambda x: x['date'].minute, axis=1)
			df_stamp['minute'] = df_stamp['minute'] // 15
			df_stamp['hour'] = df_stamp.apply(lambda x: x['date'].hour, axis=1)
			df_stamp['day'] = df_stamp.apply(lambda x: x['date'].day, axis=1)
			df_stamp['weekday'] = df_stamp.apply(lambda x: x['date'].weekday(), axis=1)
			df_stamp['month'] = df_stamp.apply(lambda x: x['date'].month, axis=1)
			data_stamp = df_stamp.drop('date', axis=1).values
		elif self.timeenc == 1:
			data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
			data_stamp = data_stamp.transpose(1, 0)

		self.data_x = data[border1:border2]
		if self.inverse:
			self.data_y = df_data.values[border1:border2]
		else:
			self.data_y = data[border1:border2]
		self.data_stamp = data_stamp

	def __getitem__(self, index):
		s_begin = index
		s_end = s_begin + self.seq_len
		r_begin = s_end - self.label_len
		r_end = r_begin + self.label_len + self.pred_len

		seq_x = self.data_x[s_begin:s_end]
		if self.inverse:
			seq_y = self.data_x[r_begin:r_end + self.label_len]
		else:
			seq_y = self.data_y[r_begin:r_end + self.label_len]
		seq_x_mark = self.data_stamp[s_begin:s_end]
		seq_y_mark = self.data_stamp[r_begin:r_end]

		return seq_x, seq_y, seq_x_mark, seq_y_mark

	def __len__(self):
		return len(self.data_x) - self.seq_len + 1

	def inverse_transform(self, data):
		return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
	pass


class Dataset_Custom(Dataset):
	pass


def data_provider(args, flag):
	data_dict = {
		'ETTh1': Dataset_ETT_hour,
		'ETTm1': Dataset_ETT_minute,
		'custom': Dataset_Custom,
	}
	Data = data_dict[args.data]
	# timef对时间轴进行embedding,其他标准化到[-0.5， 0.5]
	timeenc = 0 if args.embed != 'timeF' else 1

	if flag == 'test':
		shuffle_flag = False
		drop_last = True
		batch_size = args.batch_size
		freq = args.freq
	elif flag == 'pred':
		shuffle_flag = False
		drop_last = False
		batch_size = 1
		freq = args.freq
		Data = Dataset_Pred
	else:
		# train, valid
		shuffle_flag = True
		drop_last = True
		batch_size = args.batch_size
		freq = args.freq

	data_set = Data(
		root_path=args.root_path,
		data_path=args.data_path,
		flag=flag,
		size=[args.seq_len, args.label_len, args.pred_len],
		features=args.features,
		target=args.target,
		timeenc=timeenc,
		freq=freq
	)
	print(flag, len(data_set))
	data_loader = DataLoader(
		data_set,
		batch_size=batch_size,
		shuffle=shuffle_flag,
		drop_last=drop_last
	)
	return data_set, data_loader


class EarlyStopping:
	def __init__(self, patience=7, verbose=False, delta=0):
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.inf
		self.delta = delta

	def save_checkpoint(self, val_loss, model, path):
		if self.verbose:
			print(f'validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
		torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
		self.val_loss_min = val_loss

	def __call__(self, val_loss, model, path):
		score = - val_loss
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, path)
		elif score < self.best_score + self.delta:
			self.counter += 1
			print(f'early stopping counter : {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, path)
			self.counter = 0


def adjust_learning_rate(optimizer, epoch, args):
	if args.lradj == 'type1':
		lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
	elif args.lradj == 'type2':
		lr_adjust = {
			2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
			10: 5e-7, 15: 1e-7, 20: 5e-8
		}
	if epoch in lr_adjust.keys():
		lr = lr_adjust[epoch]
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
		print('updating learning rate to {}'.format(lr))


def visual(true, preds=None, path='pic/test.pdf'):
	plt.figure()
	plt.plot(true, label='groundtruth', linewidth=2)
	if preds is not None:
		plt.plot(preds, label='prediction', linewidth=2)
	plt.legend()
	plt.savefig(path, bbox_inches='tight')


def MAE(pred, true):
	return np.mean(np.abs(pred-true))


def MSE(pred, true):
	return np.mean((pred - true) ** 2)


def RMSE(pred, true):
	return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
	return np.mean(np.abs(pred - true) / true)


def MSPE(pred, true):
	return np.mean(np.square((pred - true) / true))


def metric(pred, true):
	mae = MAE(pred, true)
	mse = MSE(pred, true)
	rmse = RMSE(pred, true)
	mape = MAPE(pred, true)
	mspe = MSPE(pred, true)
	return mae, mse, rmse, mape, mspe


class Exp_Main():
	def __init__(self, args):
		self.args = args
		self.device = self._acquire_device()
		self.model = self._build_model().to(self.device)

	def _acquire_device(self):
		if self.args.use_gpu:
			# 指定GPU,详细说明 https://zhuanlan.zhihu.com/p/385352354
			os.environ['CUDA_VISIBLE_DEVICES'] = str(
				self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
			device = torch.device('cuda:{}'.format(self.args.gpu))
			print('use gpu: cuda:{}'.format(self.args.gpu))
		else:
			device = torch.device('cpu')
			print('use cpu')
		return device

	def _build_model(self):
		model_dict = {
			'Autoformer': Autoformer,
			'Transformer': Transformer,
			'Informer': Informer,
			'Reformer': Reformer,
			'DLinear': DLinear,
			'Nlinear': NLinear,
			'FEDformer': FEDformer,
			'TimesNet': TimesNet,
			'NBeats': NBeats,
			'PatchTST': PatchTST,
		}
		# 把model的数据float
		model = model_dict[self.args.model](self.args).float()
		if self.args.use_gpu and self.args.use_multi_gpu:
			# 多gpu训练模型(多显卡训练模型精度略微低于单显卡)
			model = nn.DataParallel(model, device_ids=self.args.device_ids)
		return model

	def _select_optimizer(self):
		model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
		return model_optim

	def _select_criterion(self):
		criterion = nn.MSELoss()
		return criterion

	def _get_data(self, flag):
		data_set, data_loader = data_provider(self.args, flag)
		return data_set, data_loader

	def vali(self, vali_data, vali_loader, criterion):
		total_loss = []
		self.model.eval()
		with torch.no_grad():
			for i, batch in enumerate(vali_loader):
				batch_x, batch_y, batch_x_mark, batch_y_mark = (
					item.float().to(self.device) for item in batch
				)
				dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
				dec_inp = torch.cat(
					[batch_y[:, :self.args.label_len, :], dec_inp],
					dim=1
				).float().to(self.device)
				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						if self.args.output_attention:
							outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
						else:
							outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

						f_dim = -1 if self.args.features == 'MS' else 0
						outputs = outputs[:, -self.args.pred_len:, f_dim:]
						batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
						loss = criterion(outputs, batch_y)
						total_loss.append(loss.item())
				else:
					if self.args.output_attention:
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
					else:
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

					f_dim = -1 if self.args.features == 'MS' else 0
					outputs = outputs[:, -self.args.pred_len:, f_dim:]
					batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
					loss = criterion(outputs, batch_y)
					total_loss.append(loss.item())

		total_loss = np.mean(total_loss)
		self.model.train()
		return total_loss

	def train(self, setting):
		train_data, train_loader = self._get_data(flag='train')
		vali_data, vali_loader = self._get_data(flag='val')
		test_data, test_loader = self._get_data(flag='test')

		path = os.path.join(self.args.checkpoints, setting)
		if not os.path.exists(path):
			os.makedirs(path)

		time_now = time.time()

		train_steps = len(train_loader)
		early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

		model_optim = self._select_optimizer()
		criterion = self._select_criterion()

		# 混合精度, 可以在不降低模型精度的情况下加速模型, https://zhuanlan.zhihu.com/p/348554267
		if self.args.use_amp:
			scaler = torch.cuda.amp.GradScaler()

		for epoch in range(self.args.train_epochs):
			iter_count = 0
			train_loss = []

			self.model.train()
			epoch_time = time.time()
			for i, batch in enumerate(train_loader):
				iter_count += 1
				model_optim.zero_grad()

				batch_x, batch_y, batch_x_mark, batch_y_mark = (
					item.float().to(self.device) for item in batch
				)
				# decoder input
				dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
				dec_inp = torch.cat(
					[batch_y[:, :self.args.label_len, :], dec_inp],
					dim=1
				).float().to(self.device)
				# encoder-decoder
				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						if self.args.output_attention:
							outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
						else:
							outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

						f_dim = -1 if self.args.features == 'MS' else 0
						outputs = outputs[:, -self.args.pred_len:, f_dim:]
						batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
						loss = criterion(outputs, batch_y)
						train_loss.append(loss.item())
				else:
					if self.args.output_attention:
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
					else:
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

					f_dim = -1 if self.args.features == 'MS' else 0
					outputs = outputs[:, -self.args.pred_len:, f_dim:]
					batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
					loss = criterion(outputs, batch_y)
					train_loss.append(loss.item())

				if (iter_count + 1) % 100 == 0:
					print('\t iters: {}, epoch: {} | loss: {:.7f}'.format(
						i+1, epoch+1, loss.item()
					))
					speed = (time.time() - time_now) / iter_count
					left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
					print('\t speed: {:.4f}s/iter; left time: {:.4f}s'.format(
						speed, left_time
					))
					iter_count = 0
					time_now = time.time()

				if self.args.use_amp:
					scaler.scale(loss).backward()
					scaler.step(model_optim)
					scaler.update()
				else:
					loss.backward()
					model_optim.step()

			# epoch打印模型数据
			print('epoch: {} cost time: {}'.format(epoch+1, time.time()-epoch_time))
			train_loss = np.mean(train_loss)
			vali_loss = self.vali(vali_data, vali_loader, criterion)
			test_loss = self.vali(test_data, test_loader, criterion)
			print('epoch: {}, steps: {} | train loss: {:.7f} vali loss: {:.7f} test loss: {:.7f}'.format(
				epoch+1, train_steps, train_loss, vali_loss, test_loss
			))

			# early_stopping防止模型过拟合
			early_stopping(vali_loss, self.model, path)
			if early_stopping.early_stop:
				print('early stopping')
				break

			# epoch调整lr
			adjust_learning_rate(model_optim, epoch+1, self.args)

		best_model_path = os.path.join(path, 'checkpoint.pth')
		self.model.load_state_dict(torch.load(best_model_path))
		return self.model

	def test(self, setting, test=0):
		test_data, test_loader = self._get_data(flag='test')
		if test:
			print('loading model')
			self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

		preds = []
		trues = []
		result_path = os.path.join(self.args.result_path, setting)
		if not os.path.exists(result_path):
			os.makedirs(result_path)

		self.model.eval()
		with torch.no_grad():
			for i, batch in enumerate(test_loader):
				batch_x, batch_y, batch_x_mark, batch_y_mark = (
					item.float().to(self.device) for item in batch
				)
				dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
				dec_inp = torch.cat(
					[batch_y[:, :self.args.label_len, :], dec_inp],
					dim=1
				).float().to(self.device)
				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						if self.args.output_attention:
							outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
						else:
							outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
				else:
					if self.args.output_attention:
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
					else:
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

				f_dim = -1 if self.args.features == 'MS' else 0
				outputs = outputs[:, -self.args.pred_len:, f_dim:]
				batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
				# batch_size, pred_len, df
				outputs = outputs.detach().cpu().numpy()
				batch_y = batch_y.detach().cpu().numpy()
				pred = outputs
				true = batch_y
				preds.append(pred)
				trues.append(true)

				if i % 20 == 0:
					# batch_size, seq_len, df
					input = batch_x.detach().cpu().numpy()
					gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
					pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
					visual(gt, pd, os.path.join(result_path, str(i)+'.pdf'))

		preds = np.concatenate(preds, axis=0)
		trues = np.concatenate(trues, axis=0)
		metric_test = metric(preds, trues)
		print(metric_test)

	def predict(self, setting, load=False):
		pred_data, pred_loader = self._get_data('pred')
		if load:
			print('loading model')
			self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

		preds = []

		self.model.eval()
		with torch.no_grad():
			for i, batch in enumerate(pred_loader):
				batch_x, batch_y, batch_x_mark, batch_y_mark = (
					item.float().to(self.device) for item in batch
				)
				dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
				dec_inp = torch.cat(
					[batch_y[:, :self.args.label_len, :], dec_inp],
					dim=1
				).float().to(self.device)
				if self.args.use_amp:
					with torch.cuda.amp.autocast():
						if self.args.output_attention:
							outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
						else:
							outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
				else:
					if self.args.output_attention:
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
					else:
						outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

				f_dim = -1 if self.args.features == 'MS' else 0
				outputs = outputs[:, -self.args.pred_len:, f_dim:]
				outputs = outputs.detach().cpu().numpy()
				pred = outputs
				preds.append(pred)

		preds = np.concatenate(preds, axis=0)
		preds = pred_data.inverse_transform(preds)
		return preds


class TokenEmbedding(nn.Module):
	def __init__(self, c_in, d_model):
		super(TokenEmbedding, self).__init__()
		self.tokenConv = nn.Conv1d(
			in_channels=c_in, out_channels=d_model,
			kernel_size=3, padding=1,
			padding_mode='circular', bias=False
		)
		# mode='fan_in'保证向前传播反差不变
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

	def forward(self, x):
		"""
		:param x: batch_size, seq_len, df
		:return:
		"""
		x = x.permute(0, 2, 1)
		# 在seq_len方向进行一维卷积
		x = self.tokenConv(x)
		x = x.permute(0, 2, 1)
		return x


class FixedEmbedding(nn.Module):
	"""
	10.6.3. 位置编码
	https://zh-v2.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html
	"""
	def __init__(self, c_in, d_model):
		super(FixedEmbedding, self).__init__()
		w = torch.zeros(c_in, d_model).float()
		w.require_grad = False
		# position在seq_len方向，div_term在d方向
		position = torch.arange(0, c_in).float().unsqueeze(1)
		div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
		w[:, 0::2] = torch.sin(position * div_term)
		w[:, 1::2] = torch.cos(position * div_term)
		self.emb = nn.Embedding(c_in, d_model)
		self.emb.weight = nn.Parameter(w, requires_grad=False)

	def forward(self, x):
		return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
	def __init__(self, d_model, embed_type='fixed', freq='h'):
		super(TemporalEmbedding, self).__init__()
		minute_size = 4
		hour_size = 24
		weekday_size = 7
		day_size = 32
		month_size = 13
		Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
		if freq == 't':
			self.minute_embed = Embed(minute_size, d_model)
		self.hour_embed = Embed(hour_size, d_model)
		self.weekday_embed = Embed(weekday_size, d_model)
		self.day_embed = Embed(day_size, d_model)
		self.month_embed = Embed(month_size, d_model)

	def forward(self, x):
		x = x.long()
		minute_x = self.minute_embed(x[:, :, 0]) if hasattr(self, 'minute_embed') else 0.0
		hour_x = self.hour_embed(x[:, :, 1])
		day_x = self.day_embed(x[:, :, 2])
		weekday_x = self.weekday_embed(x[:, :, 3])
		month_x = self.month_embed(x[:, :, 4])
		return minute_x + hour_x + day_x + weekday_x + month_x


class TimeFeatureEmbedding(nn.Module):
	def __init__(self, d_model, embed_type='timeF', freq='h'):
		super(TimeFeatureEmbedding, self).__init__()
		freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
		d_inp = freq_map[freq]
		self.embed = nn.Linear(d_inp, d_model, bias=False)

	def forward(self, x):
		return self.embed(x)


class moving_avg(nn.Module):
	def __init__(self, kernel_size, stride):
		super(moving_avg, self).__init__()
		self.kernel_size = kernel_size
		self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

	def forward(self, x):
		"""
		:param x: batch_size, seq_len, d
		:return:
		"""
		front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
		end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
		# 在seq_len方向用开头结尾的数值填充，使得滑动平均以后seq_len保持不变, kernel_size奇数
		x = torch.cat([front, x, end], dim=1)
		# 在seq_len方向滑动平均
		x = x.permute(0, 2, 1)
		x = self.avg(x)
		x = x.permute(0, 2, 1)
		return x


class series_decomp(nn.Module):
	def __init__(self, kernel_size):
		super(series_decomp, self).__init__()
		self.moving_avg = moving_avg(kernel_size, stride=1)

	def forward(self, x):
		moving_mean = self.moving_avg(x)
		res = x - moving_mean
		# res代表了周期性, moving_mean代表了趋势性
		return res, moving_mean


class series_decomp_multi(nn.Module):
	def __init__(self, kernel_size):
		super(series_decomp_multi, self).__init__()
		self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
		self.layer = nn.Linear(1, len(kernel_size))

	def forward(self, x):
		moving_mean = []
		for func in self.moving_avg:
			moving_avg = func(x)
			moving_mean.append(moving_avg.unsqueeze(dim=-1))
		moving_mean = torch.cat(moving_mean, dim=-1)
		moving_mean = torch.sum(moving_mean * nn.Softmax(dim=-1)(self.layer(x.unsqueeze(dim=-1))), dim=-1)
		res = x - moving_mean
		return res, moving_mean


