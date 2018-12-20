import tensorflow as tf

from feature_formater import format_data, get_word_vec, get_feature, add_vec2feature

data = get_feature()

word_vec = get_word_vec()

add_vec2feature(data, word_vec)

