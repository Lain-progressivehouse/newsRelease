import fastText as ft

def make_wordsfile(document_list):
	"""
	fastText学習用にテキストデータを作成する
	:param document_list: 文書リスト
	:return:
	"""
	model_text = ""
	for document in document_list["news"]:
		model_text += " ".join(document) + "\n"

	w = open("./model/for_papers/dataset.txt", "w")
	w.write(model_text)
	w.close()

model = ft.train_unsupervised(input="../model/for_papers/dataset.txt", model="skipgram", dim=200, ws=10,
                              minCount=20, loss="ns", neg=10, epoch=25, thread=40, wordNgrams=1, t=1e-3)
model.save_model("./model/for_papers/fastText.model")
# model = ft.train_unsupervised(input="./model/for_papers/dataset.txt", model="skipgram", dim=200, ws=10,
#                               minCount=20, loss="ns", neg=10, epoch=25, thread=40, wordNgrams=1, minn=3, maxn=6, t=1e-3)
# model = ft.train_unsupervised(input="./data/fastText2013to2017_dataSet_2.txt", model="skipgram", dim=200, ws=10,
#                               minCount=20, loss="ns", neg=10, epoch=25, thread=40, wordNgrams=2)
# # model = ft.train_unsupervised(input="./data/fastText2013to2017_dataSet.txt", model="skipgram", dim=100, ws=10, minCount=20, epoch=25, thread=4, wordNgrams=2)
# model.save("./model/fastText2013to2017.model")
