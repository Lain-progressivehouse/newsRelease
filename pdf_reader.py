from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


def get_pdf_strings(file):
	rsrcmgr = PDFResourceManager()
	rettxt = StringIO()
	laparams = LAParams()
	# 縦書き文字を横並びで出力する
	# laparams.detect_vertical = False
	device = TextConverter(rsrcmgr, rettxt, codec="utf-8", laparams=laparams)
	# 処理するPDFを開く
	fp = open(file, 'rb')
	interpreter = PDFPageInterpreter(rsrcmgr, device)

	# maxpages：ページ指定（0は全ページ）
	for page in PDFPage.get_pages(fp, pagenos=None, maxpages=0, password="", caching=True, check_extractable=True):
		interpreter.process_page(page)

	document = rettxt.getvalue()

	fp.close()
	device.close()
	rettxt.close()

	return document
