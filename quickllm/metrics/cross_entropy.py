# -*- coding: utf-8 -*- 
"""
    @Project ：quickllm 
    @File    ：cross_entropy.py
    @Author  ：ys
    @Time    ：2023/12/12 15:27
    添加TF2实现的categorical_cross_entropy
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def multiple_label_categorical_cross_entropy(y_true, y_pred, dtype=tf.float32, **keys):
   """copy from https://github.com/bojone/bert4keras/blob/master/bert4keras/backend.py
   多标签分类的交叉熵
   说明：
      1. y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类；
      2. 请保证y_pred的值域是全体实数，换言之一般情况下
         y_pred不用加激活函数，尤其是不能加sigmoid或者
         softmax；
      3. 预测阶段则输出y_pred大于0的类；
   """

   # XXX 强行转化类型，处理某些情况发生上下溢的问题，影响待确认
   if not dtype is None:
      y_true = tf.cast(y_true, dtype=dtype)
      y_pred = tf.cast(y_pred, dtype=dtype)

   y_pred = (1 - 2 * y_true) * y_pred
   y_pred_neg = y_pred - y_true * 1e12
   y_pred_pos = y_pred - (1 - y_true) * 1e12
   zeros = K.zeros_like(y_pred[..., :1])
   y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
   y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
   neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
   pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
   return neg_loss + pos_loss