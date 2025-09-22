import numpy as np

def get_noise_level(difficulty="easy"):
    '''Get noise parameters for different difficulty levels'''
    noise_profiles = {
        'easy': {
            'bearing_std': np.radians(0.1),
            'pixel_std_close': 0.3,
            'pixel_std_far': 1.0
        },
        'medium': {
            'bearing_std': np.radians(0.3),
            'pixel_std_close': 0.7,
            'pixel_std_far': 2.0
        },
        'hard': {
            'bearing_std': np.radians(0.5),
            'pixel_std_close': 1.0,
            'pixel_std_far': 3.0
        },
        'realistic': {
            'bearing_std': np.radians(0.2),
            'pixel_std_close': 0.5,
            'pixel_std_far': 'adaptive'
        }
    }
    return noise_profiles[difficulty]

def add_simple_adaptive_noise(true_bearing, true_pixel_size, true_range, difficulty="medium", far_range=5000):
    '''Add noise to the bearing and pixel size measurements'''
    noise_params = get_noise_level(difficulty)

    bearing_noise = np.random.normal(0, noise_params['bearing_std'])

    if noise_params['pixel_std_far'] == 'adaptive':
        noise_scaling_range = 50
        pixel_std = max(noise_params['pixel_std_close'], noise_params['pixel_std_close'] * (true_range / noise_scaling_range))
    
    else:
        close_range = 1
        t = (true_range - close_range) / (far_range - close_range)
        pixel_std = (1-t)*noise_params['pixel_std_close'] + t*noise_params['pixel_std_far']

    pixel_noise = np.random.normal(0, pixel_std)

    quantized_size = np.round(true_pixel_size)

    return true_bearing + bearing_noise, quantized_size + pixel_noise

def get_adaptive_R(predicted_range, difficulty="medium", far_range=5000):
    '''Get adaptive measurement noise covariance R based on predicted range'''
    noise_params = get_noise_level(difficulty)

    if noise_params['pixel_std_far'] == 'adaptive':
        noise_scaling_range = 50
        pixel_std = max(noise_params['pixel_std_close'], noise_params['pixel_std_close'] * (predicted_range / noise_scaling_range))
    
    else:
        close_range = 1
        t = (predicted_range - close_range) / (far_range - close_range)
        pixel_std = (1-t)*noise_params['pixel_std_close'] + t*noise_params['pixel_std_far']

    R = np.diag(np.array([noise_params['bearing_std'], pixel_std]))**2
    return R