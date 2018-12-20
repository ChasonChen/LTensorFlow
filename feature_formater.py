import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def is_send(x):
    return 0 if x == "pop_show" else 1


def merge(d_item, d_tag):
    return d_item + '-' + d_tag


def format_data():
    df = pd.read_csv('data/training.csv')
    item = df['item'].apply(lambda x: is_send(x))
    tag = df['tag']
    data = {'item': item,
            'tag': tag}
    feature = pd.DataFrame(data, columns=['item', 'tag', 'vec'])
    # feature.to_csv('data/feature.csv')
    return feature


def get_feature():
    return pd.read_csv('data/feature.csv')


MAX_WORDS_COUNT = 4


def add_vec2feature(data, word_vec):
    feature = np.empty([data.shape[0], 4 * 49])
    for index, row in data.iterrows():
        word_row = []
        try:
            words = str(row['tag']).split(' ')
            for j in range(MAX_WORDS_COUNT):
                if j < words.__len__():
                    word_row.append(word_vec[words[j]])
        except KeyError:
            print ('Match error')
        feature[index, :word_row.__len__()] = word_row

    return feature


def get_word_vec():
    file_path = 'data/glove.6B/glove.6B.50d.txt'
    s_file = open(file_path)
    word_vec = {}
    while True:
        line = s_file.readline()
        words = line.split()
        if not line:
            break
        word = words[0]
        vec = words[1:50]
        word_vec[word] = vec
    return word_vec
