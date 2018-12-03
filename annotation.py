"""
文書をクラスタリング→クラスタに属する文書の単語の頻度と単語ベクトルの大きさから内容を推測する
"""

import collections
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import pickle
import textwrap


def count_words(document_list, n=100):
	"""
	引数の文書すべての単語の出現数を返す
	:param document_list: 文書リスト
	:param n: 上位何個を返すか
	:return: {word: value}
	"""
	return collections.Counter(list(itertools.chain(*document_list))).most_common(n)


def get_vector_size(word_vec):
	"""
	ベクトルの大きさを取得
	:param word_vec: 単語ベクトル
	:return: 大きさ
	"""
	return np.linalg.norm(word_vec)


def get_cluster(word_vec, num_features=200, num_clusters=60):
	"""

	:param word_vec:
	:param num_features:
	:param num_clusters:
	:return:
	"""
	clusters = [get_vector_size(word_vec[num_features * i:num_features * (i + 1) - 1]) for i in range(0, num_clusters)]
	return clusters


def get_score(word_vec, count):
	"""
	単語ベクトルの大きさと単語の出現数でスコアを算出
	:param word_vec:
	:param count:
	:return:
	"""
	return (get_vector_size(word_vec) - 4) * count


def calc_center(positions):
	"""
	中心点を計算
	:param positions: 座標リスト
	:return: 中心点[x,y]
	"""
	return np.average(positions, axis=0)


def main(document_list, prob_wordvecs, tsne, cluster=100):
	# クラスタ数
	num_clt = cluster

	# document_list = pickle.load(open("./model/for_papers/document_list.pickle", "rb"))
	# prob_wordvecs = pickle.load(open("./model/fastText_prob_wordvecs_2", "rb"))
	# tsne = np.load("./model/tsne_2.npy")
	pred = KMeans(n_clusters=num_clt).fit_predict(tsne)

	# {クラスタ: 文書のインデックス}
	clt_dict = collections.defaultdict(list)
	for clt, pos, doc in zip(pred, tsne, document_list["news"]):
		clt_dict[clt].append([pos, doc])

	# ポジション
	annotate_positions = []
	# 単語
	annotate_words = []
	# スコア
	annotate_scores = []
	for i in range(0, num_clt):
		# (単語, 出現数)
		words = count_words([l[1] for l in clt_dict[i]])

		# 単語のスコアを取得
		scores = []
		for word in words:
			try:
				scores.append(get_score(prob_wordvecs[word[0]], word[1]))
			except:
				scores.append(0)

		print(len(scores))

		# クラスタの中心点を求める
		pos = calc_center([l[0] for l in clt_dict[i]])

		# print(words)
		# print(np.max(scores))

		# スコア合計点
		total_score = 0

		# クラスタのスコアの高い単語を取得
		topic_words = []

		for i in range(0, 3):
			# total_score += np.max(scores)
			# topic_words.append(words.pop(np.argmax(scores))[0])

			# もし12字以上なら8文字で改行+半角スペース2つを挿入
			topic_word = words.pop(np.argmax(scores))[0]
			if len(topic_word) >= 12:
				topic_word = "-\n".join(textwrap.wrap(topic_word, 8))
			topic_words.append(topic_word)

			scores.pop(np.argmax(scores))

		print(",".join(topic_words) + ": " + str(pos))
		# annotate_text.append(topic_words[0])
		annotate_words.append("\n".join(topic_words))
		annotate_positions.append(pos)
		# annotate_scores.append(total_score)

	# スコアの正規化
	# print(annotate_scores)
	# normalized_score = np.round(annotate_scores / np.linalg.norm(annotate_scores) * 10, 1)
	# print(np.round(annotate_scores / np.linalg.norm(annotate_scores) * 10, 1))

	# 作図
	doc_tsne = pd.DataFrame(tsne[:, 0], columns=["x"])
	doc_tsne["y"] = pd.DataFrame(tsne[:, 1])
	# クラスタで色分け
	# doc_tsne["class"] = list(pred)
	# 企業ごとで色分け
	doc_tsne["class"] = list(document_list["company"])
	# sns.set_style("darkgrid")
	sns.lmplot(data=doc_tsne, x="x", y="y", hue="class", legend=False, fit_reg=False, size=10)

	# アノテーション
	for pos, word in zip(annotate_positions, annotate_words):
		plt.annotate(word, xy=(pos[0], pos[1]), xytext=(0, 0), textcoords='offset points', va="center", ha="center",
		             fontsize=12)

	plt.show()


if __name__ == '__main__':
	document_list = pickle.load(open("./model/document_list2013to2017_2", "rb"))
	prob_wordvecs = pickle.load(open("./model/fastText_prob_wordvecs_2", "rb"))
	tsne = np.load("./model/tsne_2.npy")
	pred = KMeans(n_clusters=100).fit_predict(tsne)

	clt_dict = collections.defaultdict(list)
	for clt, pos, doc in zip(pred, tsne, document_list["news"]):
		clt_dict[clt].append([pos, doc])

	for i in range(0, 100):
		words = count_words([l[1] for l in clt_dict[i]])

		socres = [get_score(prob_wordvecs[word[0]], word[1]) for word in words]

		print(words[np.argmax(socres)][0])
