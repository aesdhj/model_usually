import random
import math
from operator import itemgetter
from tqdm import tqdm


"""
item_cf细节参考 项亮 《推荐系统实践》 2.4.2
"""
class ItemBaseCF():
	def __init__(self, N, K, data_path='data/ratings.csv'):
		# 找到和目标顾客已看电影最相似的N部电影，推荐最相似的K部电影
		self.n_sim_movie = N
		self.n_rec_movie = K
		self.data_path = data_path

		self.trainset = {}
		self.testset = {}
		self.movie_sim_matrix = {}
		self.movie_popular = {}
		self.movie_count = 0

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
			if random.random() < pivot:
				self.trainset.setdefault(user, {})
				self.trainset[user][movie] = rating
			else:
				self.testset.setdefault(user, {})
				self.testset[user][movie] = rating

	def cal_movie_sim(self):
		for user, movies in self.trainset.items():
			for movie in movies:
				if movie not in self.movie_popular:
					self.movie_popular[movie] = 0
				self.movie_popular[movie] += 1

		self.movie_count = len(self.movie_popular)

		for user, movies in tqdm(self.trainset.items(), desc='movie cross count'):
			for m1 in movies:
				for m2 in movies:
					if m1 == m2:
						continue
					self.movie_sim_matrix.setdefault(m1, {})
					self.movie_sim_matrix[m1].setdefault(m2, 0)
					self.movie_sim_matrix[m1][m2] += 1
					# 消除活跃用户的影响
					# self.movie_sim_matrix[m1][m2] += 1/math.log(1+len(movies))
		for m1, related_movies in tqdm(self.movie_sim_matrix.items(), desc='movie sim'):
			for m2, count in related_movies.items():
				self.movie_sim_matrix[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])

	def recommend(self, user):
		recommend_movies = {}
		watched_movies = self.trainset[user]

		for movie, rating in watched_movies.items():
			for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[0:self.n_sim_movie]:
				if related_movie in watched_movies:
					continue
				recommend_movies.setdefault(related_movie, 0)
				# 用户->商品->关联商品
				recommend_movies[related_movie] += 1.0 * w
		return sorted(recommend_movies.items(), key=itemgetter(1), reverse=True)[0:self.n_rec_movie]

	def evaluate(self):
		hit = 0
		rec_count = 0
		test_count = 0
		all_rec_movies = set()

		for user, movies in self.trainset.items():
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
	userCF = ItemBaseCF(20, 10)
	userCF.get_dataset()
	userCF.cal_movie_sim()
	userCF.evaluate()
	# 1682 6100 25220
	# precision=0.2757, recall=0.0667, coverage=0.0654

