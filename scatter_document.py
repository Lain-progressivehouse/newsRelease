import numpy as np
import seaborn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import bhtsne

from scdv import dataFrame
doc_list = dataFrame.DataFrame.get_document_list()

tsne = np.load("./model/tsne.npy")

doc_tsne = pd.DataFrame(tsne[:, 0], columns=["x"])
doc_tsne["y"] = pd.DataFrame(tsne[:, 1])
doc_tsne["class"] = list(doc_list["company"])
sns.lmplot(data=doc_tsne, x="x", y="y", hue="class", fit_reg=False, size=8)