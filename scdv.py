# 必要なライブラリをインポート
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.mixture import GaussianMixture
from scdv import dataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

num_features = 200
min_word_count = 20
num_workers = 40
context = 10
downsampling = 1e-3
num_clusters = 60

min_no = 0
max_no = 0


def get_probability_word_vectors(word2vec_model, word_centroid_map,
                                 word_centroid_prob_map, num_clusters, word_idf_dict):
	"""
	確率重み付け単語ベクトルを求める
	:param word2vec_model: 単語ベクトル
	:param word_centroid_map: [単語:その単語が属するクラスタ]の辞書
	:param word_centroid_prob_map: [単語: 単語がクラスタに属する確率]の辞書
	:param num_clusters: GMMのクラスタ数
	:param word_idf_dict: [単語:その単語のidf値]のリスト
	:return:確率重み付け単語ベクトル
	"""

	print("確率重み付き単語ベクトルを求めます...")

	prob_wordvecs = {}
	for word in word_centroid_map:  # 各単語に以下の処理を繰り返す
		prob_wordvecs[word] = np.zeros(num_clusters * num_features, dtype="float32")  # ゼロを要素とする配列を作成 クラスタ数*単語ベクトルの次元数
		for index in range(0, num_clusters):  # 各クラスタに以下の処理を繰り返す
			try:

				prob_wordvecs[word][index * num_features:(index + 1) * num_features] \
					= word2vec_model[word] * word_centroid_prob_map[word][index] * word_idf_dict[word]
			except:
				continue

	print("確率重み付き単語ベクトル取得完了")

	return prob_wordvecs


def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, word_centroid_map, dimension, num_centroids, train=False):
	"""
	ドキュメントDnについてベクトルを初期化し、Dnに含まれる単語wiinDnについてベクトルを足し合わせていき平均する
	:param prob_wordvecs: 確率重み付け単語ベクトル
	:param wordlist: 文書の単語リスト
	:param word_centroid_map: [単語:その単語が属するクラスタ]のリスト
	:param dimension: 単語ベクトルの次元数
	:param num_centroids: GMMの次元数
	:param train:
	:return: 文書に含まれる単語ごとの確率重み付け単語ベクトルを取得して足し合わせてそのベクトルの大きさで各値を割ったベクトル
	"""

	print("create cluster vector and gwbowv...")

	# This function computes SDV feature vectors.
	bag_of_centroids = np.zeros(num_centroids * dimension, dtype="float32")
	global min_no
	global max_no

	for word in wordlist:
		try:
			temp = word_centroid_map[word]
		except:
			continue
		# 文書に含まれる単語ごとの確率重み付け単語ベクトルを取得して足し合わせていく
		bag_of_centroids += prob_wordvecs[word]

	# ベクトルの大きさを求める
	norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
	# ベクトルの各値をベクトルの大きさで割る
	if norm != 0:
		bag_of_centroids /= norm

	# 最大値と最小値を保持
	if train:
		min_no += min(bag_of_centroids)
		max_no += max(bag_of_centroids)

	print("completed")

	return bag_of_centroids


def create_cluster_vector_and_gwbowv(prob_wordvecs, wordlist, dimension, num_centroids, train=False):
	"""
	ドキュメントDnについてベクトルを初期化し、Dnに含まれる単語wiinDnについてベクトルを足し合わせていき平均する
	:param prob_wordvecs: 確率重み付け単語ベクトル
	:param wordlist: 文書の単語リスト
	:param dimension: 単語ベクトルの次元数
	:param num_centroids: GMMの次元数
	:param train:
	:return: 文書に含まれる単語ごとの確率重み付け単語ベクトルを取得して足し合わせてそのベクトルの大きさで各値を割ったベクトル
	"""
	# This function computes SDV feature vectors.
	bag_of_centroids = np.zeros(num_centroids * dimension, dtype="float32")
	global min_no
	global max_no

	for word in wordlist:
		try:
			# 文書に含まれる単語ごとの確率重み付け単語ベクトルを取得して足し合わせていく
			bag_of_centroids += prob_wordvecs[word]
		except:
			continue

	# ベクトルの大きさを求める
	norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
	# ベクトルの各値をベクトルの大きさで割る
	if norm != 0:
		bag_of_centroids /= norm

	# 最大値と最小値を保持
	if train:
		min_no += min(bag_of_centroids)
		max_no += max(bag_of_centroids)

	return bag_of_centroids


def cluster_GMM(word2vec_model):
	"""
	GMMクラスタリングを行う
	:param word2vec_model: クラスタリングするモデル
	:return: クラスタ, 事後確率
	"""

	print("GMMクラスタリングを行います...")

	# GMMクラスタリング
	word_vectors = word2vec_model.wv.syn0  # word2vecのベクトルの行列
	clf = GaussianMixture(n_components=60, covariance_type="tied", init_params='kmeans', max_iter=100)
	clf.fit(word_vectors)
	# idx = clf.predict(word_vectors)  # 訓練されたモデルを使用して、Xのデータサンプルのラベルを予測します。
	# idx_prob = clf.predict_proba(word_vectors)  # データを与えられた各成分の事後確率を予測する
	idx = clf.predict([word2vec_model[v] for v in word2vec_model.wv.index2word])
	idx_prob = clf.predict_proba([word2vec_model[v] for v in word2vec_model.wv.index2word])

	print("GMMクラスタリング完了")

	return idx, idx_prob


def get_idf(document_list):
	"""
	idf値を取得する
	:param document_list: 文書リスト
	:return:
	"""

	print("idf値の取得を実行します...")

	tfv = TfidfVectorizer(dtype=np.float32)
	tfv.fit_transform(document_list)
	featurenames = tfv.get_feature_names()
	idf = tfv._tfidf.idf_

	word_idf_dict = {}
	for pair in zip(featurenames, idf):
		word_idf_dict[pair[0]] = pair[1]

	print("idf値取得完了")

	return word_idf_dict


def get_prob_wordvecs_for_fastText(fastText, document_list, cluster=60):
	clf = GaussianMixture(n_components=cluster, covariance_type="tied", init_params='kmeans', max_iter=100)
	word_vectors = [fastText.get_word_vector(w) for w in fastText.get_words()]
	clf.fit(word_vectors)
	word_centroid_map = dict(
		zip(fastText.get_words(), clf.predict([fastText.get_word_vector(w) for w in fastText.get_words()])))
	word_centroid_prob_map = dict(
		zip(fastText.get_words(), clf.predict_proba([fastText.get_word_vector(w) for w in fastText.get_words()])))
	word_idf_dict = get_idf([" ".join(v) for v in document_list["news"]])
	wordvecs = {}
	for w in fastText.get_words():
		wordvecs[w] = fastText.get_word_vector(w)
	prob_wordvecs = get_probability_word_vectors(wordvecs, word_centroid_map, word_centroid_prob_map, cluster, word_idf_dict)
	return prob_wordvecs


def get_prob_wordvecs(word2vec_model, document_list):
	"""
	確率重み付き単語ベクトルを取得する
	:param word2vec_model: word2vecのモデル
	:return: 確率重み付き単語ベクトル
	"""
	word_idf_dict = get_idf([" ".join(v) for v in document_list["news"]])

	idx, idx_prob = cluster_GMM(word2vec_model)

	word_centroid_map = dict(zip(word2vec_model.wv.index2word, idx))
	word_centroid_prob_map = dict(zip(word2vec_model.wv.index2word, idx_prob))

	prob_wordvecs = get_probability_word_vectors(word2vec_model, word_centroid_map, word_centroid_prob_map, num_clusters, word_idf_dict)

	return prob_wordvecs


def get_scdv(prob_wordvecs, document_list):
	"""
	scdvを求める
	:param prob_wordvecs: 確率重み付き単語ベクトル
	:param document_list:
	:return:
	"""
	# (文書数,GMMの次元数*word2vecの次元数)の0行列
	gwbowv = np.zeros((document_list["news"].size, num_clusters * num_features), dtype="float32")

	counter = 0

	global max_no
	global min_no

	max_no = 0
	min_no = 0

	for words in document_list["news"]:
		# 文書に含まれる単語を取得
		gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, words, num_features, num_clusters, train=True)
		counter += 1

	percentage = 0.04
	min_no = min_no * 1.0 / len(document_list["news"])
	max_no = max_no * 1.0 / len(document_list["news"])
	print("Average min: ", min_no)
	print("Average max: ", max_no)
	thres = (abs(max_no) + abs(min_no)) / 2
	thres = thres * percentage

	# 閾値よりも小さい行列の値をゼロにする
	temp = abs(gwbowv) < thres
	gwbowv[temp] = 0

	return gwbowv
