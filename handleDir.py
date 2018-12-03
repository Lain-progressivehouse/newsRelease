import os
import re
from scdv import dataFrame


def find_all_files(directory):
	"""
	引数のディレクトリ下のファイルのパスの一覧を取得する
	:param directory: ディレクトリ
	:return: ファイルのパスの一覧を取得
	"""
	for root, dirs, files in os.walk(directory):
		for file in files:
			if re.search(r"[\s\S]*?.html", file):
				yield os.path.join(root, file)
			elif re.search(r"[\s\S]*?.htm", file):
				yield os.path.join(root, file)


def handle_dir(path):
	"""
	分かち書きした文書を名前を日付にして保存する
	:param path: パス
	:return:
	"""
	df = dataFrame.DataFrame

	for file in find_all_files(path + "/Press"):
		try:
			r = open(file, 'r', encoding='shift_jis')
			document = r.read()
			r.close()
		except:
			try:
				r = open(file, 'r', encoding='utf-8')
				document = r.read()
				r.close()
			except:
				try:
					r = open(file, 'r', encoding='iso2022_jp')
					document = r.read()
					r.close()
				except:
					print(file)
					continue

		if re.search(r"20[0-9]{2}年[0-9]{1,2}月[0-9]{1,2}日", document):
			d = re.search(r"20[0-9]{2}年[0-9]{1,2}月[0-9]{1,2}日", document)  # 日付を取得
		else:
			continue
		date = re.sub(r"[年月/]", "-", d.group(0))  # 年月/を-に変換
		date = re.sub(r"日", "", date)  # 日を削除
		date = re.sub(r"-([0-9])-", r"-0\1-", date)  # 20yy-m-ddを20yy-0m-ddに変換
		date = re.sub(r"-([0-9])\Z", r"-0\1", date)  # 20yy-mm-dを20yy-0m-0dに変換
		document = df.html_tag_remover(document)  # 分かち書きにして空白区切り

		if os.path.exists(path + "/textInfo/" + date + ".txt"):
			w = open(is_same_name_file(path + "/textInfo/" + date, 1), "w")
		else:
			w = open(path + "/textInfo/" + date + ".txt", "w")
		w.write(document)
		w.close()


def is_same_name_file(path, n):
	"""
	path_n.txtを返す. すでに存在するならpath_n+1.txt....
	:param path: ファイル名
	:param n: 回数
	:return: 保存するパス
	"""
	if os.path.exists(path + "_" + str(n) + ".txt"):
		return is_same_name_file(path, n + 1)
	else:
		return path + "_" + str(n) + ".txt"
