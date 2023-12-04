import re
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
    cleaned_description = ml.get_and_clean_data()[:1000]
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'[^A-Za-z]', ' ', s))
    cleaned_description = cleaned_description.apply(lambda s: re.sub(r'\s+', ' ', s))
    print(cleaned_description)

    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))

    stop_set = set(stopwords.words())
    sw_removed_description = tokenized_description.apply(lambda s: set(s) - stop_set)
    sw_removed_description = sw_removed_description.apply(lambda s: [word for word in s if len(word) > 2])

    # caching
    concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)

    # stem
    stemmed_description = sw_removed_description.apply(lambda s: [stem_cache[w] for w in s])
    print(stemmed_description)

    cv = CountVectorizer(analyzer=lambda x: x)
    X = cv.fit_transform(stemmed_description)
    print(pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out()))
    print(X.tocsr()[0, :])

    XX = X.toarray()
    print(np.shape(np.matmul(X.toarray(), X.toarray().T)))
    print(timeit.timeit(lambda: np.matmul(XX, XX.T), number=1))

    print(np.shape(X * X.T))
    print(timeit.timeit(lambda: X * X.T, number=1))

    print(timeit.timeit(lambda: np.matmul(XX, XX.T), number=3) / 3)
    print(timeit.timeit(lambda: X.todok() * X.T.todok(), number=3) / 3)
    print(timeit.timeit(lambda: X.tolil() * X.T.tolil(), number=3) / 3)
    print(timeit.timeit(lambda: X.tocoo() * X.T.tocoo(), number=3) / 3)
    print(timeit.timeit(lambda: X.tocsc() * X.T.tocsc(), number=3) / 3)
