import tensorflow as tf
cfg=tf.compat.v1.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1
    # session_inter_op_thread_pool=1,
    # use_per_session_threads=False,
    # device_count=[1]
    )
serialized = cfg.SerializeToString()
print(list(map(hex, serialized)))