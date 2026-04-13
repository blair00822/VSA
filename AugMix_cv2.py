import cv2
import numpy as np
import torch


class AugmentCV:
    def __init__(self, image_size=100):
        self.image_size = image_size
        self.augmentation_names = [
            'autocontrast', 
            'equalize', 
            'solarize', 
            'brightness',
            'contrast',
            # 'color',
            'rotate', 
            'shear_x', 
            'shear_y', 
            'translate_x', 
            'translate_y', 
            'flip',
            # 'cutout',
        ] 
        self.augmentations = [getattr(self, f'_{name}') for name in self.augmentation_names]
        
    def __call__(self, img, param_cache=None):
        img = (img * 255).astype(np.uint8) if img.max() < 1 else img.astype(np.uint8)
        
        if param_cache is None:
            param_cache = self._generate_param_cache(mixture_width=3, aug_severity=5)

        # mix = torch.zeros_like(img)
        mix = np.zeros_like(img, dtype=np.float32)
        m = param_cache['mix_weight']
        ws = param_cache['ws']
        ops_list = param_cache['ops']

        for i, ops in enumerate(ops_list):
            img_aug = img.copy()
            for op_name in ops:
                op = getattr(self, f'_{op_name}')
                level = param_cache['params'][op_name]
                rand = param_cache['rand_flags'][op_name]
                # level = param_cache['params'][(i, op_name)]
                # rand = param_cache['rand_flags'][(i, op_name)]
                img_aug = op(img_aug, level, rand)
            mix += ws[i] * img_aug.astype(np.float32)

        mixed = (1 - m) * img.astype(np.float32) + m * mix
        return mixed
    
    def _generate_param_cache(self, mixture_width, aug_severity):
        aug_severity = np.random.uniform(0.1, aug_severity)
        param_cache = {
            'ws': np.float32(np.random.dirichlet([1] * mixture_width)),
            'mix_weight': np.float32(np.random.beta(1, 1)), # Beta(1, 1) ≡ Uniform(0, 1)
            'params': {},  
            'ops': [],    
            'rand_flags': {} # _rotate, _shear, _translate
        }

        for i in range(mixture_width):
            depth = np.random.randint(1, 4) ##
            ops = np.random.choice(self.augmentation_names, size=depth) ## replace = False
            param_cache['ops'].append(ops)

            for op_name in ops:
                if op_name not in param_cache['params']:
                    param_cache['params'][op_name] = self._sample_param(op_name, aug_severity)
                    param_cache['rand_flags'][op_name] = np.random.rand()
                # key = (i, op_name)
                # param_cache['params'][key] = self._sample_param(op_name, aug_severity)
                # param_cache['rand_flags'][key] = np.random.rand()
        return param_cache
    
    def _sample_param(self, op_name, level):
        
        param_map = {
            'autocontrast': lambda: None,
            'equalize': lambda: None,
            'rotate':      lambda: self.int_parameter(level, 30),
            'solarize':    lambda: self.int_parameter(level, 256),
            'shear_x':     lambda: self.float_parameter(level, 0.3),
            'shear_y':     lambda: self.float_parameter(level, 0.3),
            'translate_x': lambda: self.int_parameter(level, self.image_size / 3),
            'translate_y': lambda: self.int_parameter(level, self.image_size / 3),
            'brightness':  lambda: self.float_parameter(level, 1.8) + 0.1,
            'contrast':    lambda: self.float_parameter(level, 1.8) + 0.1,
            'color':       lambda: self.float_parameter(level, 1.8) + 0.1,
            'flip':        lambda: None,  
            'cutout':      lambda: self.int_parameter(level, self.image_size * 0.5),
            'noise':       lambda: self.float_parameter(level, 10),       # std ∈ [0.1, 5.0]
            'blur':        lambda: self.int_parameter(level, 20),           # kernel size ∈ [1, 3, 5, 7, 9]

        }

        if op_name not in param_map:
            raise ValueError(f"Unknown op_name '{op_name}' in _sample_param")

        return param_map[op_name]()

    def int_parameter(self, level, maxval):
        return int(level * maxval / 10)

    def float_parameter(self, level, maxval):
        return float(level) * maxval / 10
    
    def _autocontrast(self, img, *_): return cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    def _equalize(self, img, *_): 
        gray_img = img[:, :, 0] if len(img.shape) == 3 else img
        aug_img = cv2.equalizeHist(gray_img)
        aug_img = np.stack([aug_img] * 3, axis=2)
        return aug_img
        
    def _rotate(self, img, degrees, rand):
        if rand > 0.5:
            degrees = -degrees
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), degrees, 1)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _solarize(self, img, threshold, *_):
        threshold = 256 - threshold
        return np.where(img < threshold, img, 255 - img).astype(np.uint8)
    
    def _shear_x(self, img, level, rand):
        if rand > 0.5:
            level = -level
        h, w = img.shape[:2]
        M = np.float32([[1, level, 0], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
  
    def _shear_y(self, img, level, rand):
        if rand > 0.5:
            level = -level
        h, w = img.shape[:2]
        M = np.float32([[1, 0, 0], [level, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _translate_x(self, img, level, rand):
        if rand > 0.5:
            level = -level
        M = np.float32([[1, 0, level], [0, 1, 0]])
        h, w = img.shape[:2]
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def _translate_y(self, img, level, rand):
        if rand > 0.5:
            level = -level
        M = np.float32([[1, 0, 0], [0, 1, level]])
        h, w = img.shape[:2]
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    
    def _brightness(self, img, alpha, *_):
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    def _contrast(self, img, level, *_):
        gray_img = img[:, :, 0] if len(img.shape) == 3 else img
        mean = gray_img.mean()
        aug_img = cv2.addWeighted(gray_img, level, gray_img, 0, mean * (1 - level))
        aug_img = np.stack([aug_img] * 3, axis=2)
        return aug_img
    
    def _color(self, img, level, *_):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= level
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _flip(self, img, *_): 
        return cv2.flip(img, 1)
    
    def _cutout(self, img, level, *_):
        h, w = img.shape[:2]
        
        cy = h // 2
        cx = w // 2

        x1 = np.clip(cx - level, 0, w)
        x2 = np.clip(cx + level, 0, w)
        y1 = np.clip(cy - level, 0, h)
        y2 = np.clip(cy + level, 0, h)

        img[y1:y2, x1:x2] = 0  
        return img
    
    def _noise(self, img, std, *_):
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy_img = img.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img
    
    def _blur(self, img, kernel_size, *_):
        k = max(1, kernel_size) 
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), 0)





