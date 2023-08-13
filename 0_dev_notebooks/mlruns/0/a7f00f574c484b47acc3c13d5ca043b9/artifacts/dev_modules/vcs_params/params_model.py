# Model hyperparameters.


# pooling type.
POOL_TYPE = "average"
# pooling kernel size.
POOL_KERNEL_SIZE = 2
# stride reduction factor
POOL_STRIDE = 2


# convolution layers.
# Lx = layer "x".
L0_N_FILTER = 2
L0_KERNEL_SIZE = 3

L1_N_FILTER = 4
L1_KERNEL_SIZE = 3

L2_N_FILTER = 8
L2_KERNEL_SIZE = 3

L3_N_FILTER = 16
L3_KERNEL_SIZE = 3

L4_N_FILTER = 32
L4_KERNEL_SIZE = 3 


# decision threshold.
THRESHOLD_DECISION = 0.5

# summary to logger.
TRAIN_HYPER = dict(
                   # pooling.
                   Pool_type = POOL_TYPE,
                   Pool_size = POOL_KERNEL_SIZE,
                   Pool_stride = POOL_STRIDE,
                   
                   # conv layers.
                   N0_n_filter = L0_N_FILTER,
                   N0_size = L0_KERNEL_SIZE,

                   N1_n_filter = L1_N_FILTER,                   
                   N1_size = L1_KERNEL_SIZE,

                   N2_n_filter = L2_N_FILTER,                   
                   N2_size = L2_KERNEL_SIZE,
                   
                   N3_n_filter = L3_N_FILTER,                   
                   N3_size = L3_KERNEL_SIZE,
                   
                   N4_n_filter = L4_N_FILTER,
                   N4_size = L4_KERNEL_SIZE,
                   
                   # decision threshold.
                   Threshold_decision = THRESHOLD_DECISION)
