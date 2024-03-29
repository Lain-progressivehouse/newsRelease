from scdv import dataFrame
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import bhtsne
import datetime
import time
from sklearn.cluster import KMeans
import collections
from scdv import annotation
import logging
from operator import itemgetter

"""
散布図作成用
"""


def create_docvec_scatter(matrix, document_list):
	"""
	文書ベクトルの散布図を作成する
	:param matrix: 行列
	:return:
	"""
	tsne = bhtsne.tsne(matrix.astype(sp.float64), dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1)
	doc_tsne = pd.DataFrame(tsne[:, 0], columns=["x"])
	doc_tsne["y"] = pd.DataFrame(tsne[:, 1])
	doc_tsne["class"] = list(document_list["company"])
	sns.set_style("darkgrid")
	sns.lmplot(data=doc_tsne, x="x", y="y", hue="class", fit_reg=False, size=10)
	# 図を表示
	plt.show()


def create_wordvec_scatter(matrix, vocab):
	"""
	単語ベクトルの散布図を作成する
	:param matrix: 単語ベクトルの行列
	:param vocab: 単語リスト
	:return:
	"""
	tsne = bhtsne.tsne(matrix.astype(sp.float64), dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1)
	plt.figure(figsize=(32, 24))  # 図のサイズ
	plt.scatter(tsne[0:241], tsne[0:241, 1])

	count = 0
	for label, x, y in zip(vocab, tsne[0:241], tsne[0:241, 1]):
		count += 1
		if count < 0:
			continue

		plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

		if count == 500:
			break

	plt.show()


def retern_stock_scatter(tsne, document_list):

	stocks = dataFrame.DataFrame.get_stocks()
	companys = document_list["company"]
	dates = document_list["date"]

	floating_value = []
	for company, date in zip(companys, dates):
		# 変動値のリストを取得
		floating_value.append(get_return(company, date, stocks))

	floating_value = np.array(floating_value)

	median = np.median(floating_value)

	doc_tsne = pd.DataFrame(tsne[:, 0], columns=["x"])
	doc_tsne["y"] = pd.DataFrame(tsne[:, 1])
	doc_tsne["floating"] = floating_value > 0
	sns.set_style("darkgrid")
	sns.lmplot(data=doc_tsne, x="x", y="y", hue="floating", fit_reg=False, size=10)
	# 図を表示
	plt.show()

	for news, is_hight_return in zip(document_list["news"], floating_value > 0):
		if is_hight_return:
			print(" ".join(news))

def multidimensional_retern(prob_wordvecs, document_list, pred, cluster=50):
	# pred = KMeans(n_clusters=cluster).fit_predict(scdv)

	# 製品系: 0, 経営系: 1, その他: 2, 技術系: 3
	num = [1, 1, 3, 3, 3, 0, 3, 0, 2, 0,
	       3, 0, 1, 1, 3, 0, 0, 0, 1, 0,
	       3, 2, 0, 0, 0, 2, 2, 0, 2, 1,
	       2, 3, 0, 0, 1, 2, 0, 2, 3, 3,
	       0, 2, 2, 2, 2, 0, 2, 2, 2, 3]

	stocks = dataFrame.DataFrame.get_stocks()
	companys = document_list["company"]
	dates = document_list["date"]

	# リターンをリストで取得
	return_value = []
	# 企業, 日付, クラスタ
	com_day_clt = []
	for company, date, clt in zip(companys, dates, pred):
		# 同じ企業が同じ日に，同じクラスタに属する記事を出している場合除外
		if [company, date, clt] in com_day_clt:
			return_value.append((-1))
			continue
		com_day_clt.append([company, date, clt])
		# 変動値のリストを取得
		return_value.append(get_return(company, date, stocks, is_abs=False))

	a = np.array([[c, r] for c, r in zip(pred, return_value) if r != -1 and r > -0.7])
	doc_tsne = pd.DataFrame(list(map(int, a[:, 0])), columns=["cluster"])
	doc_tsne["return"] = pd.DataFrame(a[:, 1])
	doc_tsne["num"] = [num[i] for i in list(map(int, a[:, 0]))]
	sns.set_style("darkgrid")
	plt.figure(figsize=(15, 8))
	sns.boxplot(data=doc_tsne, x="cluster", y="return", hue="num", dodge=False)
	plt.show()


	return_value = np.array(return_value)

	# {クラスタ: [リターン]}
	clt_dict = collections.defaultdict(list)
	for clt, value in zip(pred, return_value):
		clt_dict[clt].append(value)

	# リターン
	annotate_return = []

	for i in range(0, cluster):
		# クラスタ内のリターンの平均を求める
		# annotate_return.append(np.average([l for l in clt_dict[i] if l != -1]))
		annotate_return.append(np.std([l for l in clt_dict[i] if l != -1 and l > -0.7]))



	a = np.array([[c, r] for c, r in zip(pred, return_value) if r != -1])
	# doc_tsne = pd.DataFrame(list(range(0, cluster)), columns=["cluster"])
	# doc_tsne["Volatility"] = pd.DataFrame(annotate_return)
	# doc_tsne["num"] = num
	# sns.set_style("darkgrid")
	# sns.set_context("poster")
	# plt.figure(figsize=(15, 8))
	# sns.barplot(data=doc_tsne, x="cluster", y="Volatility", hue="num", dodge=False)
	# plt.show()

	# {クラスタ: 文書のインデックス}
	clt_dict = collections.defaultdict(list)
	for clt, doc in zip(pred, document_list["news"]):
		clt_dict[clt].append(doc)

	# 単語
	annotate_words = []

	for i in range(0, cluster):
		# (単語, 出現数)
		words = annotation.count_words([l for l in clt_dict[i]])

		# 単語のスコアを取得
		scores = []
		for word in words:
			try:
				scores.append(annotation.get_score(prob_wordvecs[word[0]], word[1]))
			except:
				scores.append(0)

			# クラスタのスコアの高い単語を取得
			topic_words = []

		for i in range(0, 5):
			try:
				topic_word = words.pop(np.argmax(scores))[0]
			except:
				topic_word = "null"
			topic_words.append(topic_word)
			try:
				scores.pop(np.argmax(scores))
			except:
				continue

		annotate_words.append(",".join(topic_words))

	# for i, word, stock in sorted(zip(range(0, cluster), annotate_words, annotate_return), key=lambda x:x[2], reverse=True):
	# 	print("{:.2%}".format(stock) + ": " + word)

	csv = ""
	for i, word, stock in zip(range(0, cluster), annotate_words, annotate_return):
		print("[" + str(i) + "] " + "{:.2%}".format(stock) + ": " + word)
		csv += str(i) + "," + word + "\n"

	return csv

def multidimensional_production_increase_rate(prob_wordvecs, document_list, pred, cluster=50):
	# pred = KMeans(n_clusters=cluster).fit_predict(scdv)

	stocks = dataFrame.DataFrame.get_stocks()
	companys = document_list["company"]
	dates = document_list["date"]

	# リターンをリストで取得
	return_value = []
	# 企業, 日付, クラスタ
	com_day_clt = []
	for company, date, clt in zip(companys, dates, pred):
		# 同じ企業が同じ日に，同じクラスタに属する記事を出している場合除外
		if [company, date, clt] in com_day_clt:
			return_value.append((-1))
			continue
		com_day_clt.append([company, date, clt])
		# 変動値のリストを取得
		return_value.append(get_production_increase_rate(company, date, stocks))

	a = np.array([[c, r] for c, r in zip(pred, return_value) if r != -1])
	doc_tsne = pd.DataFrame(list(map(int, a[:, 0])), columns=["x"])
	doc_tsne["y"] = pd.DataFrame(a[:, 1])
	sns.set_style("darkgrid")
	plt.figure(figsize=(15, 8))
	sns.boxplot(data=doc_tsne, x="x", y="y")
	plt.show()

	return_value = np.array(return_value)

	# {クラスタ: [リターン]}
	clt_dict = collections.defaultdict(list)
	for clt, value in zip(pred, return_value):
		clt_dict[clt].append(value)

	# リターン
	annotate_return = []

	for i in range(0, cluster):
		# クラスタ内のリターンの平均を求める
		annotate_return.append(np.average([l for l in clt_dict[i] if l != -1]))

	# {クラスタ: 文書のインデックス}
	clt_dict = collections.defaultdict(list)
	for clt, doc in zip(pred, document_list["news"]):
		clt_dict[clt].append(doc)

	# 単語
	annotate_words = []

	for i in range(0, cluster):
		# (単語, 出現数)
		words = annotation.count_words([l for l in clt_dict[i]])

		# 単語のスコアを取得
		scores = []
		for word in words:
			try:
				scores.append(annotation.get_score(prob_wordvecs[word[0]], word[1]))
			except:
				scores.append(0)

			# クラスタのスコアの高い単語を取得
			topic_words = []

		for i in range(0, 10):
			try:
				topic_word = words.pop(np.argmax(scores))[0]
			except:
				topic_word = "null"
			topic_words.append(topic_word)
			try:
				scores.pop(np.argmax(scores))
			except:
				continue

		annotate_words.append(", ".join(topic_words))

	for i, word, stock in sorted(zip(range(0, cluster), annotate_words, annotate_return), key=lambda x:x[2], reverse=True):
		print("{:.2%}".format(stock) + ": " + word)


def retern_stock_cluster(tsne, document_list, cluster=50):
	# Kmeansでクラスタリング
	pred = KMeans(n_clusters=cluster).fit_predict(tsne)

	stocks = dataFrame.DataFrame.get_stocks()
	companys = document_list["company"]
	dates = document_list["date"]

	# リターンをリストで取得
	return_value = []
	# 企業, 日付, クラスタ
	com_day_clt = []
	for company, date, clt in zip(companys, dates, pred):
		# 同じ企業が同じ日に，同じクラスタに属する記事を出している場合除外
		if [company, date, clt] in com_day_clt:
			return_value.append((-1))
			continue
		com_day_clt.append([company, date, clt])
		# 変動値のリストを取得
		return_value.append(get_return(company, date, stocks))

	return_value = np.array(return_value)

	# {クラスタ: (x,y), リターン}
	clt_dict = collections.defaultdict(list)
	for clt, pos, value in zip(pred, tsne, return_value):
		clt_dict[clt].append([pos, value])

	# ポジション
	annotate_positions = []
	# リターン
	annotate_return = []

	for i in range(0, cluster):
		# クラスタの中心点を求める
		annotate_positions.append(np.average([l[0] for l in clt_dict[i]], axis=0))

		# クラスタ内のリターンの平均を求める
		annotate_return.append(np.average([l[1] for l in clt_dict[i] if l[1] != -1]))

		# クラスタ内のリターンの中央値を求める
		# annotate_return.append(np.max([l[1] for l in clt_dict[i] if l[1] != -1]))

	doc_tsne = pd.DataFrame(tsne[:, 0], columns=["x"])
	doc_tsne["y"] = pd.DataFrame(tsne[:, 1])
	doc_tsne["class"] = list(document_list["company"])
	sns.set_style("darkgrid")
	sns.lmplot(data=doc_tsne, x="x", y="y", hue="class", legend=False, fit_reg=False, size=10)

	# アノテーション
	for pos, rtn in zip(annotate_positions, annotate_return).sort():
		plt.annotate("{:.2%}".format(rtn), xy=(pos[0], pos[1]), xytext=(0, 0), textcoords='offset points', va="center", ha="center",
		             fontsize=12)

	# 図を表示
	plt.show()

	# -------------------

	# リターンをリストで取得
	return_value = []
	for company, date in zip(companys, dates):
		# 変動値のリストを取得
		return_value.append(get_return(company, date, stocks, is_abs=False))

	return_value = np.array(return_value)

	# {クラスタ: (x,y), リターン}
	clt_dict = collections.defaultdict(list)
	for clt, pos, value in zip(pred, tsne, return_value):
		clt_dict[clt].append([pos, value])

	# リターン
	annotate_return = []

	for i in range(0, cluster):
		# クラスタ内のリターンの平均を求める
		annotate_return.append(np.average([l[1] for l in clt_dict[i] if l[1] != -1]))

		# クラスタ内のリターンの中央値を求める
		# annotate_return.append(np.max([l[1] for l in clt_dict[i] if l[1] != -1]))

	doc_tsne = pd.DataFrame(tsne[:, 0], columns=["x"])
	doc_tsne["y"] = pd.DataFrame(tsne[:, 1])
	doc_tsne["class"] = list(document_list["company"])
	sns.set_style("darkgrid")
	sns.lmplot(data=doc_tsne, x="x", y="y", hue="class", legend=False, fit_reg=False, size=10)

	# アノテーション
	for pos, rtn in zip(annotate_positions, annotate_return):
		plt.annotate("{:.2%}".format(rtn), xy=(pos[0], pos[1]), xytext=(0, 0), textcoords='offset points', va="center", ha="center",
		             fontsize=12)

	# 図を表示
	plt.show()


def get_return(company, date, stocks, is_abs=True):
	"""
	会社名と日付から直近のリターンを返す
	:param company: 会社名
	:param date: 日付
	:param stocks: 株価データ
	:return: リターン
	"""
	# 株価情報取得

	# 日付をdatetime型に変換
	date = datetime.datetime.strptime(date, "%Y-%m-%d")

	start = -1
	end = -1

	# dateの当日が存在するかどうか
	if date in stocks[company].index:
		# 存在するならその日の始値を取得
		# line = stocks[company][date == stocks[company].index]
		# start = line.get("始値").item()

		# 当日以外ならその日以前の直近の終値を取得
		lines = stocks[company][date - datetime.timedelta(days=30):date - datetime.timedelta(days=1)]
		# サイズが0でないなら
		if lines.size != 0:
			line = lines.tail(1)
			# start = line.get("終値").item()
			start = line.get("終値調整値").item()

	else:
		# 当日以外ならその日以前の直近の終値を取得
		lines = stocks[company][date - datetime.timedelta(days=30):date - datetime.timedelta(days=1)]
		# サイズが0でないなら
		if lines.size != 0:
			line = lines.tail(1)
			# start = line.get("終値").item()
			start = line.get("終値調整値").item()

	# dateの次の日以降の終値を取得
	lines = stocks[company][date + datetime.timedelta(days=1):date + datetime.timedelta(days=31)]
	if lines.size != 0:
		line = lines.head(1)
		# end = line.get("終値").item()
		end = line.get("終値調整値").item()

	# 値が存在するなら差の絶対値を返す. 存在しないなら-1を返す
	if start != -1 and end != -1:
		if is_abs:
			return abs(start - end) / start
		else:
			return (end - start) / start
	else:
		return -1

def get_beta_value(prob_wordvecs, document_list, pred, cluster=50):
	stocks = dataFrame.DataFrame.get_stocks()
	companys = document_list["company"]
	dates = document_list["date"]

	# リターンをリストで取得
	return_value = []
	# 日経平均のリターンをリストで取得
	nikkei_value = []
	# 企業, 日付, クラスタ
	com_day_clt = []
	for company, date, clt in zip(companys, dates, pred):
		# 同じ企業が同じ日に，同じクラスタに属する記事を出している場合除外
		if [company, date, clt] in com_day_clt:
			return_value.append((-1))
			continue
		com_day_clt.append([company, date, clt])
		# 変動値のリストを取得
		return_value.append(get_return(company, date, stocks, is_abs=False))
		nikkei_value.append(get_return(company, date, stocks, is_abs=False))

	return_value = np.array(return_value)
	nikkei_value = np.array(return_value)

	# {クラスタ: [リターン, 日経]}
	clt_dict = collections.defaultdict(list)
	for clt, value, nikkei in zip(pred, return_value, nikkei_value):
		clt_dict[clt].append([value, nikkei])

	# 共分散
	beta_list = []

	for i in range(0, cluster):
		# クラスタ内のリターンの平均を求める
		a = np.cov([[l[0] for l in clt_dict[i] if l[0] != -1 and l[1] != -1], [l[1] for l in clt_dict[i] if l[0] != -1 and l[1] != -1]])
		b = np.var([l[1] for l in clt_dict[i] if l[0] != -1 and l[1] != -1])
		beta_list.append(a / b)

	# {クラスタ: 文書のインデックス}
	clt_dict = collections.defaultdict(list)
	for clt, doc in zip(pred, document_list["news"]):
		clt_dict[clt].append(doc)

	# 単語
	annotate_words = []

	for i in range(0, cluster):
		# (単語, 出現数)
		words = annotation.count_words([l for l in clt_dict[i]])

		# 単語のスコアを取得
		scores = []
		for word in words:
			try:
				scores.append(annotation.get_score(prob_wordvecs[word[0]], word[1]))
			except:
				scores.append(0)

			# クラスタのスコアの高い単語を取得
			topic_words = []

		for i in range(0, 10):
			try:
				topic_word = words.pop(np.argmax(scores))[0]
			except:
				topic_word = "null"
			topic_words.append(topic_word)
			try:
				scores.pop(np.argmax(scores))
			except:
				continue

		annotate_words.append(", ".join(topic_words))

	for i, word, stock in zip(range(0, cluster), annotate_words, beta_list):
		print(str(stock) + ": " + word)

def get_production_increase_rate(company, date, stocks):
	"""
	出来高増加率を返す
	:param company:
	:param date:
	:param stocks:
	:return:
	"""
	# 日付をdatetime型に変換
	date = datetime.datetime.strptime(date, "%Y-%m-%d")

	start = -1
	end = -1

	# dateの当日が存在するかどうか
	if date in stocks[company].index:
		# 存在するならその日の始値を取得
		# line = stocks[company][date == stocks[company].index]
		# start = line.get("始値").item()

		# 当日以外ならその日以前の直近の終値を取得
		lines = stocks[company][date - datetime.timedelta(days=30):date - datetime.timedelta(days=1)]
		# サイズが0でないなら
		if lines.size != 0:
			line = lines.tail(1)
			# start = line.get("終値").item()
			start = line.get("出来高").item()

	else:
		# 当日以外ならその日以前の直近の終値を取得
		lines = stocks[company][date - datetime.timedelta(days=30):date - datetime.timedelta(days=1)]
		# サイズが0でないなら
		if lines.size != 0:
			line = lines.tail(1)
			# start = line.get("終値").item()
			start = line.get("出来高").item()

	# dateの次の日以降の終値を取得
	# lines = stocks[company][date + datetime.timedelta(days=1):date + datetime.timedelta(days=31)]
	lines = stocks[company][date:date + datetime.timedelta(days=31)]
	if lines.size != 0:
		line = lines.head(1)
		# end = line.get("終値").item()
		end = line.get("出来高").item()

	# 値が存在するなら差の絶対値を返す. 存在しないなら-1を返す
	if start != -1 and end != -1:
		return (end - start) / start

	else:
		return -1

