import numpy as np
import scipy.spatial.distance as dis
import collections


def cos_sim(v1, v2):
	"""
	cos類似度を求める
	:param v1: ベクトル1
	:param v2: ベクトル2
	:return: cos類似度
	"""
	return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	# return dis.cosine(v1, v2)


def most_similarity(matrix, index, n=5):
	"""
	行列でcos類似度が高い順で出力する
	:param matrix: 行列
	:param index: インデックス
	:param n: 出力する数
	:return: インデックスのリスト
	"""
	cos_sim_list = {}
	count = 0

	for v in matrix:
		cos = cos_sim(matrix[index], v)
		if index == count:
			cos = 0
		cos_sim_list[count] = cos
		count += 1

	count = 0
	idx = []
	for k, v in sorted(cos_sim_list.items(), key=lambda x: -x[1]):
		if count == n:
			break
		print(str(k) + ": " + str(v))
		idx.append(k)
		count += 1

	return idx


def view_similarity_document(document_list, tsne, index):
	counter = []
	for i in index:
		print(str(tsne[i]) + ": " + str(" ".join(document_list["news"][i])))
		counter.extend(document_list["news"][i])

	print(collections.Counter(counter).most_common(100))


def most_similarity_value(matrix, value, n=5):
	cos_sim_list = {}
	count = 0

	for v in matrix:
		cos = cos_sim(value, v)
		cos_sim_list[count] = cos
		count += 1

	count = 0
	for k, v in sorted(cos_sim_list.items(), key=lambda x: -x[1]):
		if count == n:
			break
		print(str(k) + ": " + str(v))
		count += 1


def most_similarity_for_fastText(model, word, n=5):
	word_vector = model.get_word_vector(word)

	candidates = {}

	for w in model.get_words():
		if word == w:
			continue
		s = cos_sim(word_vector, model.get_word_vector(w))

		candidates[w] = s

	count = 0
	for k, v in sorted(candidates.items(), key=lambda x: -x[1]):
		if count == n:
			break
		print(str(k) + ": " + str(v))
		count += 1


def nearest_distance(document_list, matrix, position, n=5):
	"""
	二次元に次元削除したベクトルを指定した位置から近い順で出力
	:param document_list: 文書
	:param matrix: 二次元行列
	:param position: 位置
	:param n: 出力する数
	:return: インデックス
	"""
	distance_list = {}
	count = 0

	for v in matrix:
		distance = np.linalg.norm(np.array(v) - np.array(position))
		distance_list[count] = distance
		count += 1

	count = 0
	idx = []
	for k, v in sorted(distance_list.items(), key=lambda x: x[1]):
		if count == n:
			break
		print(str(k) + ": " + str(matrix[k]))
		print(document_list["date"][k] + ": " + " ".join(document_list["news"][k]))
		count += 1
		idx.append(k)

	return idx

# def most_similarity(vector, matrix, document_list, n=5):
# 	sort_cos_sim = most_similarity(vector, matrix, n=n)
# 	return [(document_list[cs[0]], cs[1]) for cs in sort_cos_sim]
