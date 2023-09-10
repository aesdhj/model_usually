import pandas as pd
import torch
from torch import nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from operator import itemgetter


"""
矩阵分解细节参考 https://zhongqiang.blog.csdn.net/article/details/108173885
"""
class SVD(nn.Module):
	def __init__(self, num_users, num_items, mean, embedding_size):
		super(SVD, self).__init__()
		self.user_emb = nn.Embedding(num_users, embedding_size)
		self.item_emb = nn.Embedding(num_items, embedding_size)
		# 用户偏差
		self.user_bias = nn.Embedding(num_users, 1)
		# 物品偏差
		self.item_bias = nn.Embedding(num_items, 1)
		# 全局平均分
		self.mean = nn.Parameter(torch.tensor([mean], dtype=torch.float32), requires_grad=False)

		self.user_emb.weight.data.uniform_(0, 0.005)
		self.user_bias.weight.data.uniform_(-0.01, 0.01)
		self.item_emb.weight.data.uniform_(0, 0.005)
		self.item_bias.weight.data.uniform_(-0.01, 0.01)

	def forward(self, u_id, i_id):
		"""
		:param u_id: (batch_size,)
		:param i_id: (batch_size,)
		:return:
		"""
		u = self.user_emb(u_id)
		i = self.item_emb(i_id)
		b_u = self.user_bias(u_id).squeeze()
		b_i = self.item_bias(i_id).squeeze()
		return (u * i).sum(dim=1) + b_u + b_i + self.mean


class SVDDataset(Dataset):
	def __init__(self, u_id, i_id, rating):
		self.u_id = u_id
		self.i_id = i_id
		self.rating = rating

	def __getitem__(self, index):
		return self.u_id[index], self.i_id[index], self.rating[index]

	def __len__(self):
		return len(self.rating)


class MF():
	def __init__(self, K, data_path='data/ratings.csv'):
		self.n_rec_movie = K
		self.data_path = data_path

		self.trainset = {}
		self.traindf = []
		self.testset = {}

	def load_file(self):
		with open(self.data_path, 'r') as file:
			for i, line in enumerate(file):
				# 跳过首行标题
				if i == 0:
					continue
				yield line.strip('\r\n')

	def get_dataset(self, pivot=0.75):
		for line in self.load_file():
			user, movie, rating, timestamp = line.split(',')
			user = int(user)
			movie = int(movie)
			rating = float(rating)
			if random.random() < pivot:
				self.trainset.setdefault(user, {})
				self.trainset[user][movie] = rating
				self.traindf.append([user, movie, rating])
			else:
				self.testset.setdefault(user, {})
				self.testset[user][movie] = rating

	def mf_cal(self, embed_size, batch_size, lr, wd, epochs):
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

		df = pd.read_csv(self.data_path)
		num_users = df['userId'].max() + 1
		num_items = df['movieId'].max() + 1
		self.movie_count = len(df[['movieId']].drop_duplicates())
		# range(num_items)里面不全是movie_id
		self.movie_id = set(df['movieId'])
		self.traindf = np.array(self.traindf)
		mean = self.traindf[:, 2].mean()
		self.svd = SVD(num_users, num_items, mean, embed_size).to(device)

		train_set = SVDDataset(
			self.traindf[:, 0].astype(np.int32),
			self.traindf[:, 1].astype(np.int32),
			self.traindf[:, 2].astype(np.float32)
		)
		train_dataloader = DataLoader(train_set, batch_size=batch_size)
		optimizer = torch.optim.Adam(params=self.svd.parameters(), lr=lr, weight_decay=wd)
		loss_func = torch.nn.MSELoss()

		for epoch in tqdm(range(epochs), desc='epoch'):
			self.svd.train()
			total_loss, total_len = 0.0, 0
			for items in train_dataloader:
				x_u, x_i, y = (item.to(device) for item in items)
				y_pred = self.svd(x_u, x_i)
				loss = loss_func(y_pred, y)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				total_loss += loss.item() * len(y)
				total_len += len(y)

			if (epoch+1) % 25 == 0:
				print(f'epoch={epoch+1}, mse_loss={total_loss/total_len:.4f}')

	def recommend(self, user):
		recommend_movies = {}
		watched_movies = self.trainset[user]
		self.svd.eval()
		with torch.no_grad():
			# (num_users, embed_size)
			user_embed = self.svd.user_emb.weight.data
			# (num_items, embed_size)
			item_embed = self.svd.item_emb.weight.data
			# (num_users, 1)
			user_bias = self.svd.user_bias.weight.data
			# (num_items,1)
			item_bias = self.svd.item_bias.weight.data
			mean = self.svd.mean.data
			# (num_items,)
			user_rating = user_embed.mm(item_embed.permute(1, 0))[user] + user_bias[user] + item_bias.squeeze() + mean
			user_rating = user_rating.to('cpu').numpy().tolist()
			for movie in range(len(user_rating)):
				if movie in watched_movies:
					continue
				# in 的花费时间 set和序列长度无关，list和序列长度有关
				if movie in self.movie_id:
					recommend_movies[movie] = user_rating[movie]
		return sorted(recommend_movies.items(), key=itemgetter(1), reverse=True)[0:self.n_rec_movie]

	def evaluate(self):
		hit = 0
		rec_count = 0
		test_count = 0
		all_rec_movies = set()

		for user, movies in tqdm(self.trainset.items(), desc='evaluate'):
			test_movies = self.testset.get(user, {})
			rec_movies = self.recommend(user)
			for movie, w in rec_movies:
				if movie in test_movies:
					hit += 1
				all_rec_movies.add(movie)
			rec_count += len(rec_movies)
			test_count += len(test_movies)

		print(hit, rec_count, test_count)
		precision = hit / rec_count
		recall = hit / test_count
		coverage = len(all_rec_movies) / self.movie_count
		print(f'precision={precision:.4f}, recall={recall:.4f}, coverage={coverage:.4f}')


if __name__ == '__main__':
	mf = MF(10)
	mf.get_dataset()
	mf.mf_cal(100, 1024, 5e-4, 1e-5, 100)
	mf.evaluate()
	# 455 6100 25485
	# precision=0.0746, recall=0.0179, coverage=0.0923














