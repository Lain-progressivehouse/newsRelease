import pickle

with open('./model/for_papers/document_list.pickle', 'rb') as f:
	d = pickle.load(f)

with open('./model/for_papers/document_list.pickle', 'wb') as f:
	pickle.dump("オブジェクト", f)
