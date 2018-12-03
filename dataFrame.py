import re
import MeCab
import pandas as pd
import glob
import os
from tqdm import tqdm
import urllib
from unicode_script import unicode_script_map
from unicode_script.unicode_script import ScriptType


class DataFrame(object):
	"""
	文書の操作を行うクラス
	"""

	@staticmethod
	def html_tag_remover(document):
		"""
		HTML文書からヘッダー, フッター, その他HTMLタグを取り除く
		:param document: htmlタグを除去する文書
		:return: htmlタグが除去された文書
		"""
		# document = re.sub(r"<!-----side_navi----->[\s\S]*?<!-----end_side_navi----->", "", document)  # inputタグ内除去

		document = re.sub(r"<!--[\s\S]*?-->", "", document)  # コメント除去
		document = re.sub(r"<style[\s\S]*?</style>", "", document)  # スタイルシートタグ内除去
		document = re.sub(r"<script[\s\S]*?</script>", "", document)  # スクリプトタグ内除去
		document = re.sub(r"<noscript[\s\S]*?</noscript>", "", document)  # スクリプトタグ内除去
		document = re.sub(r"<button[\s\S]*?</button>", "", document)  # ボタンタグ内除去
		document = re.sub(r"<input[\s\S]*?>", "", document)  # inputタグ内除去

		document = re.sub(r"pr@jp\.tdk\.com", "", document)  # spanタグ内除去
		document = re.sub(r"ニュースセンター", "", document)  # spanタグ内除去
		document = re.sub(r"<h5 class=\"text13\">報道関係者の問い合わせ先</h5>[\s\S]*?</div>", "", document)  # spanタグ内除去

		document = re.sub(r"<[aA] [\s\S]*?</[aA]>", "", document)  # リンク内除去
		document = re.sub(r"<footer[\s\S]*?</footer>", "", document)  # フッター除去
		document = re.sub(r"<header[\s\S]*?</header>", "", document)  # ヘッダー除去
		document = re.sub(r"<[^>]*?>", "", document)  # HTMLタグ除去
		document = re.sub(r"&[a-zA-Z]+?;", " ", document)  # htmlの文字参照除去
		document = re.sub(r"\s[\s]+", " ", re.sub("[\n\t]", " ", document))  # 改行とタブをスペースに変換し, 複数スペースを1つのスペースに変換する

		return document.lower()

	@staticmethod
	def document_to_wordlist(document, remove_stopwords=False):
		"""
		引数の文書を分かち書きにした単語リストに変換する
		:param document: 分かち書きする文書
		:param remove_stopwords: ストップワードを取り除くかどうか
		:return: 単語リスト
		"""
		document = document.replace('\xc2\xa0', ' ')

		stoptxt = ""
		try:
			r = open("./scdv/stopwords.txt", 'r')
			stoptxt = r.read()
			r.close()
		except:
			pass

		stopwords = [a for a in filter(("").__ne__, stoptxt.split("\n"))]

		# tokenizer = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
		# result = tokenizer.parse(document).replace("\u3000", "").replace("\n", "")
		# # result = re.sub(
		# # 	r'[◆©～－｜‘’0123456789０１２３４５６７８９・〔〕▼！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？､、。・･,./『』【】「」→←○]+',
		# # 	"", result)
		#
		# doc = ""
		# for c in result:
		# 	if unicode_script_map.get_script_type(c) != ScriptType.U_Common or c == "\u30fc" or c == "\u0020":
		# 		doc += c
		# result = doc
		#
		# wordlist = result.split(" ")
		# wordlist = filter(("").__ne__, wordlist)

		tokenizer = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
		tokenizer.parse(" ")
		node = tokenizer.parseToNode(document)
		keywords = []
		while node:
			if node.feature.split(",")[0] == u"名詞":
				keywords.append(node.surface)
			elif node.feature.split(",")[0] == u"形容詞":
				keywords.append(node.feature.split(",")[6])
			elif node.feature.split(",")[0] == u"動詞":
				keywords.append(node.feature.split(",")[6])
			node = node.next

		wordlist = []
		for word in keywords:
			word = re.sub(
				r'[◆©～－｜‘’0123456789０１２３４５６７８９・〔〕▼！＠＃＄％＾＆\-|\\＊\“（）＿■×※⇒—●(：〜＋=)／*&^%$#@!~`){}…\[\]\"\'\”:;<>?＜＞？､、。・･,./『』【】「」→←○]+',
				"", word)
			if word in stopwords:
				continue
			wordlist.append(word)

		wordlist = filter(("").__ne__, wordlist)

		return wordlist

	@staticmethod
	def get_document_list(remove_stopwords=False):
		"""
		文書リストをpandasで取得する
		:param remove_stopwords: ストップワードを取り除くかどうか
		:return: ["企業","日付",[単語リスト]]
		"""
		document_list = pd.DataFrame(columns=["company", "date", "news"])
		# dirlist = ["canon", "epson", "fujitsu", "hitachi", "j-display", "kyocera", "mitsubishielectric", "nidec",
		#            "panasonic", "ricoh", "sharp", "sony", "tdk"]
		dirlist = ["mitsubishielectric"]

		for company in dirlist:
			path = "./data/" + company + "/textInfo/*.txt"
			files = glob.glob(path)  # 配列で"../i/textInfo/*.txt"のpathを取得

			for file in tqdm(files):
				# ファイル名を取得
				file_name = os.path.basename(file)

				# ファイル名から日付を取得
				date = re.match(r"20[0-9]{2}-[0-9]{1,2}-[0-9]{1,2}", file_name).group(0)

				year = int(re.match(r"20[0-9]{2}", date).group(0))

				if year > 2017 or year < 2013:
					continue

				r = open(file, "r")
				document = r.read()
				r.bishiclose()

				# 分かち書きして単語リストとして取得
				word_list = DataFrame.document_to_wordlist(document, remove_stopwords=remove_stopwords)
				word_list = [word for word in word_list]

				t = pd.Series([company, date, word_list], index=document_list.columns)
				document_list = document_list.append(t, ignore_index=True)

		return document_list

	@staticmethod
	def get_stocks():
		dirlist = ["canon", "epson", "fujitsu", "hitachi", "j-display", "kyocera", "mitsubishielectric", "nidec",
		           "panasonic", "ricoh", "sharp", "sony", "tdk"]

		stock = {}
		for company in dirlist:
			path = "./data/" + company + "/stock/*.csv"
			files = glob.glob(path)  # 配列で"../company/stock/*.csv"のpathを取得


			data = {}
			for file in files:
				# ファイル名を取得
				file_name = os.path.basename(file)

				brand_name = re.findall(r"[0-9]{4}", file_name)[0]
				year = re.findall(r"[0-9]{4}", file_name)[1]


				with open(file, "r", encoding="cp932") as f:
					next(f)
					df = pd.read_csv(f)
					df["日付"] = pd.to_datetime(df["日付"])
					df.set_index("日付", inplace=True)
					data[year] = df
					f.close()

			stock[company] = pd.concat(data.values())

		return stock