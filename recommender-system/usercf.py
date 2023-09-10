import random
import math
from operator import itemgetter
from tqdm import tqdm


"""
user_cf细节参考 项亮 《推荐系统实践》 2.4.1
"""
class UserBaseCF():
	def __init__(self, N, K, data_path='data/ratings.csv'):
		# 找到和目标用户最相似的N个用户，推荐K个最相似的电影
		self.n_sim_user = N
		self.n_rec_movie = K
		self.data_path = data_path

		self.trainset = {}
		self.testset = {}
		self.user_sim_matrix = {}
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
				
	def cal_user_sim(self):
		movie_user = {}
		for user, movies in self.trainset.items():
			for movie in movies:
				if movie not in movie_user:
					movie_user[movie] = set()
				movie_user[movie].add(user)
		self.movie_count = len(movie_user)
		
		# 计算用户相似性		
		for movie, users in tqdm(movie_user.items(), desc='users cross count'):
			for u in users:
				for v in users:
					if u == v:
						continue
					self.user_sim_matrix.setdefault(u, {})
					self.user_sim_matrix[u].setdefault(v, 0)
					self.user_sim_matrix[u][v] += 1
					# 消除热门商品的影响
					# self.user_sim_matrix[u][v] += 1 / math.log(1+len(users))
		for u, related_users in tqdm(self.user_sim_matrix.items(), desc='users sim'):
			for v, count in related_users.items():
				self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainset[u]) * len(self.trainset[v]))
				
	def recommend(self, user):
		recommend_movies = {}
		watched_movies = self.trainset[user]

		for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0: self.n_sim_user]:
			for movie in self.trainset[v]:
				if movie in watched_movies:
					continue
				recommend_movies.setdefault(movie, 0)
				# 用户->关联用户->关联用户的商品
				recommend_movies[movie] += wuv * 1.0
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
	userCF = UserBaseCF(20, 10)
	userCF.get_dataset()
	userCF.cal_user_sim()
	userCF.evaluate()
	# 1801 6100 25092
	# precision=0.2952, recall=0.0718, coverage=0.0432
