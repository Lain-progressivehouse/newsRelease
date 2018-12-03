from scdv import dataFrame
import numpy as np
import pickle
import seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import bhtsne


def plain_word2vec_document_vector(sentence, word2vec_model, num_features):
	bag_of_centroids = np.zeros(num_features, dtype="float32")

	for word in sentence:
		try:
			temp = word2vec_model[word]
		except:
			continue
		bag_of_centroids += temp

	bag_of_centroids = bag_of_centroids / len(sentence)

	return bag_of_centroids

def plain_fastText_document_vector(sentence, fastText, num_features):
	bag_of_centroids = np.zeros(num_features, dtype="float32")

	for word in sentence:
		try:
			temp = fastText.get_word_vector(word)
		except:
			continue
		bag_of_centroids += temp

	bag_of_centroids = bag_of_centroids / len(sentence)

	return bag_of_centroids


def word_vec_average(document_list, word2vec_model):
	counter = 0
	num_features = 200
	plainDocVec_all = np.zeros((document_list["news"].size, num_features), dtype="float32")

	for sentence in document_list["news"]:
		plainDocVec_all[counter] = plain_word2vec_document_vector(sentence, word2vec_model, num_features)
		counter += 1


	tsne = bhtsne.tsne(plainDocVec_all.astype(sp.float64), dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1)
	doc_tsne = pd.DataFrame(tsne[:, 0], columns=["x"])
	doc_tsne["y"] = pd.DataFrame(tsne[:, 1])
	doc_tsne["class"] = list(document_list["company"])
	sns.lmplot(data=doc_tsne, x="x", y="y", hue="class", fit_reg=False, size=10)

	plt.show()