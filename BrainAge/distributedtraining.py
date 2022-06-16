# pylint: disable=invalid-name, redefined-outer-name
"""
Cosa fa questo modulo
"""
import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
