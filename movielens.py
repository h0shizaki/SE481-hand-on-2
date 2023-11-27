import timeit

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
# from scikit-learn import get_feature_names_out

import ml

if __name__ == '__main__':
    cleaned_description = ml.get_and_clean_data()
    cleaned_description = cleaned_description.iloc[:1000]
    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
    stop_set = set(stopwords.words())
    sw_removed_description = tokenized_description.apply(lambda s: [word for word in s if word not in stop_set])
    sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])

    ps = PorterStemmer()
    stemmed_description = sw_removed_description.apply(lambda s: [ps.stem(w) for w in s])

    cv = CountVectorizer(analyzer=lambda x: x)
    X = cv.fit_transform(stemmed_description)
    print(X.tocsr()[0, :])

    print(pd.DataFrame(X.toarray()))

    # XX = X.toarray()
    # print(np.shape(np.matmul(X.toarray(), X.toarray().T)))

    # print(timeit.timeit(lambda: np.matmul(XX, XX.T), number=3)/3)

