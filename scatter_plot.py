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
	# dirlist = ["canon", "epson", "fujitsu", "hitachi", "j-display", "kyocera", "mitsubishielectric", "nidec",
	#            "panasonic", "ricoh", "sharp", "sony", "tdk"]

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



def retern_stock_scatter_com(tsne, document_list, company):
	stocks = dataFrame.DataFrame.get_stocks()
	companys = document_list["company"]
	dates = document_list["date"]

	floating_value = []
	for company, date in zip(companys, dates):
		# 変動値のリストを取得
		floating_value.append(get_return(company, date, stocks))


	doc = document_list[document_list["company"] == company]
	tsne_com = tsne[document_list["company"] == company]

	floating_value = []
	for date in doc["date"]:
		floating_value.append(get_return(company, date, stocks))

	floating_value = np.array(floating_value)
	median = np.median(floating_value)


	doc_tsne = pd.DataFrame(tsne_com[:, 0], columns=["x"])
	doc_tsne["y"] = pd.DataFrame(tsne_com[:, 1])
	doc_tsne["floating"] = floating_value > median
	sns.set_style("darkgrid")
	sns.lmplot(data=doc_tsne, x="x", y="y", hue="floating", fit_reg=False, size=10)
	# 図を表示
	plt.show()


def get_return(company, date, stocks):
	"""
	会社名と日付から直近のリターンを返す
	:param company: 会社名
	:param date: 日付
	:param stocks: 株価データ
	:return: リターン
	"""
	# 株価情報取得
	# stocks = dataFrame.DataFrame.get_stocks()

	# 日付をdatetime型に変換
	date = datetime.datetime.strptime(date, "%Y-%m-%d")

	start = -1
	end = -1

	# dateの当日が存在するかどうか
	if date in stocks[company].index:
		# 存在するならその日の始値を取得
		line = stocks[company][date == stocks[company].index]
		start = line.get("始値").item()
	else:
		# 当日以外ならその日以前の直近の終値を取得
		lines = stocks[company][date - datetime.timedelta(days=30):date]
		# サイズが0でないなら
		if lines.size != 0:
			line = lines.tail(1)
			start = line.get("終値").item()

	# dateの次の日以降の終値を取得
	lines = stocks[company][date + datetime.timedelta(days=1):date + datetime.timedelta(days=31)]
	if lines.size != 0:
		line = lines.head(1)
		end = line.get("終値").item()

	# 値が存在するなら差の絶対値を返す. 存在しないなら-1を返す
	if start != -1 and end != -1:
		# return abs(start - end) / start
		return (end - start) / start
	else:
		return -1
