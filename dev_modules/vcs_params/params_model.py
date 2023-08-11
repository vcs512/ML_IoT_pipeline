# Model hyperparameters.


# convolution kernel size.
KERNEL_CONV_SIZE = 3    
# base number of filters.
N_FILTER_BASE = 2

# pooling kernel size.
KERNEL_POOL_SIZE = 2
# stride reduction factor
STRIDE_POOL = 2

# decision threshold.
THRESHOLD_DECISION = 0.5

# summary to logger.
TRAIN_HYPER = dict(kernel_conv_size = KERNEL_CONV_SIZE,    
                   n_filter_base = N_FILTER_BASE,
                   kernel_pool_size = KERNEL_POOL_SIZE,
                   stride_pool = STRIDE_POOL,
                   threshold_decision = THRESHOLD_DECISION)
