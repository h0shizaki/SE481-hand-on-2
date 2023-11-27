import string

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd


def get_and_clean_data():
    data = pd.read_csv('resource/software_developer_united_states_1971_20191023_1.csv')
    description = data['job_description']
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    cleaned_description = cleaned_description.apply(
        lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    cleaned_description = cleaned_description.drop_duplicates()
    return cleaned_description


def simple_tokenize(data):
    cleaned_description = data.apply(lambda s: [x.strip() for x in s.split()])
    return cleaned_description


# Tokenize the job description
def parse_job_description():
    cleaned_description = get_and_clean_data()
    cleaned_description = simple_tokenize(cleaned_description)
    return cleaned_description


def inverse_indexing(parsed_description):
    sw_set = set(stopwords.words()) - {'c'}
    no_sw_description = parsed_description.apply(lambda x: [w for w in x if w not in sw_set])
    ps = PorterStemmer()
    stemmed_description = no_sw_description.apply(lambda x: set([ps.stem(w) for w in x]))
    all_unique_term = list(set.union(*stemmed_description.to_list()))
    invert_idx = {}
    for s in all_unique_term:
        invert_idx[s] = set(stemmed_description.loc[stemmed_description.apply(lambda x: s in x)].index)
    return invert_idx


def search(invert_idx, query):
    ps = PorterStemmer()
    processed_query = [s.lower() for s in query.split()]
    stemmed = [ps.stem(s) for s in processed_query]
    matched = list(set.intersection(*[invert_idx[s] for s in stemmed]))
    return matched


if __name__ == '__main__':
    parsed_description = parse_job_description()
    invert_idx = inverse_indexing(parsed_description)
    query = 'java kafka'
    matched = search(invert_idx, query)
    print(parsed_description.loc[matched].apply(lambda x: ' '.join(x)).head().to_markdown())
