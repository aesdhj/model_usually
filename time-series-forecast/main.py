import random
import argparse
from utils import *
import warnings
warnings.filterwarnings('ignore')


def main():
	fix_seed = 2022
	random.seed(fix_seed)
	torch.manual_seed(fix_seed)
	np.random.seed(fix_seed)

	parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

	# basic config
	parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
	parser.add_argument(
		'--model', type=str, required=True, default='DLinear',
		help='model name, options: [Autoformer, Informer, Transformer, DLinear, NLinear, FEDformer, TimesNet, NBeats, Reformer, PatchTST]')

	# data loader
	parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
	parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
	parser.add_argument('--root_path', type=str, default='data/', help='root path of the data file')
	parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
	parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
	parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
	parser.add_argument('--checkpoints', type=str, default='checkpoints/', help='location of model checkpoints')
	parser.add_argument('--result_path', type=str, default='result/', help='path of result')

	# forecasting task
	parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
	parser.add_argument('--label_len', type=int, default=48, help='start token length')
	parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

	# model define
	parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
	parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
	parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
	parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
	parser.add_argument('--factor', type=int, default=1, help='attn factor')
	parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
	parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
	parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
	parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
	parser.add_argument('--activation', type=str, default='gelu', help='activation')
	parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
	parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
	parser.add_argument('--c_out', type=int, default=7, help='output size')
	parser.add_argument('--d_layers', type=int, default=1, help='num decoder layers')
	parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
	parser.add_argument('--distil', action='store_false', default=True, help='whether to use distilling in encoder, using this argument means not using distilling')

	# reformer
	parser.add_argument('--bucket_size', type=int, default=4, help='for reformer')
	parser.add_argument('--n_hashes', type=int, default=4, help='for reformer')

	# DLinear
	parser.add_argument('--individual', action='store_false', default=False, help='DLinear: a linear layer for each variate(channel) individually')

	# FEDformer
	parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
	parser.add_argument('--mode_select', type=str, default='random', help='for FEDformer, there are two mode selection method, options: [random, low]')
	parser.add_argument('--version', type=str, default='Fourier', help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')

	# TimesNet
	parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
	parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')

	#  NBeats
	parser.add_argument('--stack_types', type=list, default=['trend', 'seasonality', 'generic'], help='for NBeats')
	parser.add_argument('--thetas_dim', type=list, default=[2, 8, 3], help='for NBeats')
	parser.add_argument('--share_weights_in_stack', type=bool, default=False, help='for NBeats')
	parser.add_argument('--nb_blocks_per_stack', type=int, default=3, help='for NBeats')
	parser.add_argument('--hidden_layer_units', type=int, default=1024, help='for NBeats')
	parser.add_argument('--nb_harmonics', default=None, help='None: seasonal block thetas dim=forecast')
	parser.add_argument('--share_thetas', type=bool, default=True, help='for NBeats')

	# PatchTST
	parser.add_argument('--patch_len', type=int, default=16, help='for PatchTST')
	parser.add_argument('--stride', type=int, default=8, help='for PatchTST')

	# optimization
	parser.add_argument('--itr', type=int, default=2, help='experiment times')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
	parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
	parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
	parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
	parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
	parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
	parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

	# gpu
	parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
	parser.add_argument('--gpu', type=int, default=0, help='gpu')
	parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
	parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multi gpus')

	args = parser.parse_args()
	# notebook args = parser.parse_known_args()[0]

	args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
	if args.use_gpu and args.use_multi_gpu:
		args.device_ids = [int(id_) for id_ in args.devices.split(',')]

	print('Args in experiment:')
	print(args)

	Exp = Exp_Main
	setting = '{}_{}'.format(args.model, args.data)
	print(setting)
	if args.is_training:
		exp = Exp(args)
		exp.train(setting)
		exp.test(setting)
		if args.do_predict:
			exp.predict(setting, True)
		torch.cuda.empty_cache()
	else:
		exp = Exp(args)
		exp.test(setting, test=1)
		torch.cuda.empty_cache()


if __name__ == '__main__':
	main()

