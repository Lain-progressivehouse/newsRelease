"""
doc2vecを学習
"""
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def learn_doc2vec(document_list):
	num_features = 200  # 次元数
	min_word_count = 20  # n回未満登場する単語を破棄
	num_workers = 40  # 複数のスレッドで処理
	context = 10  # 学習に使う前後の単語数
	downsampling = 1e-3  # 単語を無視する頻度

	trainings = [TaggedDocument(sentence, tags=company) for sentence, company in zip(document_list["news"], ["doc" + str(x) for x in list(range(0,document_list["news"].size))])]

	doc2vec_model = Doc2Vec(documents=trainings, workers=num_workers, hs=0, dm=0, negative=10, epochs=25,
	                        vector_size=num_features, min_count=min_word_count, window=context, sample=downsampling, seed=1)

	document_list.rain
	# doc2vec_model.init_sims(replace=True)

	return doc2vec_model