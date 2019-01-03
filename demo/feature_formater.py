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


def get_train_data():
    data = get_feature()
    word_vec = get_word_vec()

    feature = []
    tag = []
    for index, row in data.iterrows():
        word_row = np.zeros([MAX_WORDS_COUNT, 49], dtype=np.float32)
        try:
            words = str(row['tag']).split(' ')
            for j in range(MAX_WORDS_COUNT):
                if j < words.__len__():
                    word_row[j] = np.array(word_vec[words[j]], dtype=np.float32).reshape(1, 49)
                else:
                    word_row[j] = np.zeros([1, 49], dtype=np.float32)

        except KeyError:
            err = KeyError

        feature.append(word_row)

        tag_row = np.zeros(2, dtype=np.float32)
        y = [row['item']][0]
        if y is 0:
            tag_row[0] = 1
        else:
            tag_row[1] = 1

        tag.append(tag_row)
    return np.array(feature, dtype=np.float32), np.array(tag, dtype=np.float32)


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
