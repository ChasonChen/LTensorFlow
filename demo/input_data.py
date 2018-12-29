import tensorflow as tf
import pandas as pd

from feature_formater import format_data, get_word_vec, get_feature, add_vec2feature
from trainner import add_layer

def WORD_VEC():
    df = get_feature()
    word_vec = get_word_vec()
    feature = add_vec2feature(df, word_vec)
    return feature
