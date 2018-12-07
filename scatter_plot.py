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

"""
散布図作成用
"""


def create_docvec_scatter(matrix):
	"""
	文書ベクトルの散布図を作成する
	:param matrix: 行列
	:return:
	"""
	document_list = dataFrame.DataFrame.get_document_list()

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


def retern_stock_cluster(tsne, document_list, cluster=50):
	# Kmeansでクラスタリング
	pred = KMeans(n_clusters=cluster).fit_predict(tsne)

	stocks = dataFrame.DataFrame.get_stocks()
	companys = document_list["company"]
	dates = document_list["date"]

	# リターンをリストで取得
	return_value = []
	for company, date in zip(companys, dates):
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
	for pos, rtn in zip(annotate_positions, annotate_return):
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
	# stocks = dataFrame.DataFrame.get_stocks()
	d = date

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
		lines = stocks[company][date - datetime.timedelta(days=30):date]
		# サイズが0でないなら
		if lines.size != 0:
			line = lines.tail(1)
			# start = line.get("終値").item()
			start = line.get("終値調整値").item()

	else:
		# 当日以外ならその日以前の直近の終値を取得
		lines = stocks[company][date - datetime.timedelta(days=30):date]
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
