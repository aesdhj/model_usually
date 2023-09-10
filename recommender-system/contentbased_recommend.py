import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from functools import reduce
import warnings
from collections import Counter
from pprint import pprint

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


def get_movie_dataset():
	tags = pd.read_csv('data/ml-latest-small/all-tags.csv').dropna()
	tags = tags.groupby('movieId', as_index=False)['tag'].agg(list)
	movies = pd.read_csv('data/ml-latest-small/movies.csv')
	movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
	movies = movies.merge(tags, on='movieId', how='left')
	# tag和genres合并
	movies['tag'] = movies.apply(lambda x: x['genres'] + x['tag'] if x['tag'] is not np.nan else x['genres'], axis=1)
	# 对所有tag都小写
	movies['tag'] = movies['tag'].apply(lambda x: [item.lower() for item in x])
	return movies


def create_movie_profile(movie_topn_tag=30):
	# gensim的基础入门，https://zhuanlan.zhihu.com/p/357276272
	movie_dataset = get_movie_dataset()
	tags = list(movie_dataset['tag'])
	dct = Dictionary(tags)
	corpus = [dct.doc2bow(line) for line in tags]
	model = TfidfModel(corpus)
	movie_profile = []
	for i, (index, row) in enumerate(movie_dataset.iterrows()):
		# 对应movie的每个tag的tfidf得分
		vec = model[corpus[i]]
		# 对应movie的topn tfidf得分的tag
		top_tag_weights = sorted(vec, key=lambda x: x[1], reverse=True)[:movie_topn_tag]
		top_tag_weights = dict(map(lambda x: (dct[x[0]], x[1]), top_tag_weights))
		# movie的分类权重设置为1.0
		for g in row['genres']:
			top_tag_weights[g.lower()] = 1.0
		top_tag = list(top_tag_weights)
		movie_profile.append([row['movieId'], row['title'], top_tag, top_tag_weights])

	movie_profile = pd.DataFrame(movie_profile, columns=['movieId', 'title', 'profile', 'weights'])
	return movie_profile


def create_inverted_table():
	# movie -> tag:weight, 这个movie有多少tag,weight分别是多少
	movie_profile = create_movie_profile()
	# tag -> movie:weight，这个tag在多少movie里面，weight分别是多少
	inverted_table = {}
	for index, row in movie_profile.iterrows():
		for tag, weight in row['weights'].items():
			inverted_table.setdefault(tag, [])
			inverted_table[tag].append((row['movieId'], weight))
	return inverted_table


def create_user_profile(user_topn_tag=50):
	user_profile = {}
	movie_profile = create_movie_profile()
	watch_record = pd.read_csv('data/ml-latest-small/ratings.csv')
	watch_record = watch_record.groupby('userId', as_index=False)['movieId'].agg(list)
	for index, row in watch_record.iterrows():
		# user对应看过的movie的profile
		record_movie_profile = movie_profile[movie_profile['movieId'].isin(row['movieId'])]
		# 合并看过电影的tag
		# reduce的基础入门， https://blog.csdn.net/WMM_123456/article/details/103294840
		counter = Counter(reduce(lambda x, y: x + y, list(record_movie_profile['profile'])))
		interest_tag = counter.most_common(user_topn_tag)
		max_count = interest_tag[0][1]
		# 对tag_count进行归一化，表示user_tag的兴趣程度
		interest_tag = [(tag, round(count/max_count, 4)) for tag, count in interest_tag]
		user_profile[row['userId']] = interest_tag
	return user_profile


def recommend(user_topn=10):
	# user 对应不同tag的兴趣(user看过的电影的tag的计数归一化系数)
	user_profile = create_user_profile()
	# tag 在不同电影里面的重要程度(tag的td_idf值)
	# 以tag为中间过渡，来推荐电影
	inverted_table = create_inverted_table()
	result_table = {}
	for user, tag_weight in user_profile.items():
		for tag, weight in tag_weight:
			related_movies = inverted_table[tag]
			for movie, related_weight in related_movies:
				result_table.setdefault(user, {})
				result_table[user].setdefault(movie, 0)
				# result_table[user][movie] += weight		# 只关注用户兴趣
				# result_table[user][movie] += related_weight		# 只关注tag和movie的关联度
				result_table[user][movie] += weight * related_weight		# 两者都考虑
		result_table[user] = sorted(result_table[user].items(), key=lambda x: x[1], reverse=True)[:user_topn]
	pprint(result_table)
	return result_table


"""
细节 https://zhongqiang.blog.csdn.net/article/details/111311830
"""
if __name__ == '__main__':
	recommend()


