# Core AI model for Sign Language Video Recognition
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
labels = {
    0: 'book',
    1: 'chair',
    2: 'clothes',
    3: 'computer',
    4: 'drink',
    5: 'drum',
    6: 'family',
    7: 'football',
    8: 'go',
    9: 'hat',
    10: 'hello',
    11: 'kiss',
    12: 'like',
    13: 'play',
    14: 'school',
    15: 'street',
    16: 'table',
    17: 'university',
    18: 'violin',
    19: 'wall'
}
class SLVR(object):
    def __init__(self, weights: str, img_size: int, threshold: float):
        self.path_model = weights
        self.img_size = img_size
        self.dim = (self.img_size, self.img_size)
        self.threshold = threshold
        self.model = self.init_weights(self.path_model)

    def init_weights(self, weights: str):
        model = load_model(weights)
        print(F"Loaded SLVR model successfully")
        return model

    def predict(self, frame_buffer):
        gloss_show = 'Word: none'
        frame_buffer_resh = frame_buffer.reshape(1, *frame_buffer.shape)
        predictions = self.model.predict(frame_buffer_resh)[0]
        # extract the best prediction
        best_pred_idx = np.argmax(predictions)
        acc_best_pred = predictions[best_pred_idx]

        if acc_best_pred > self.threshold:
            gloss = labels[best_pred_idx]
            gloss_show = "Word: {: <3}  {:.2f}% ".format(gloss, acc_best_pred * 100)
            return True, gloss_show
        
        return False, gloss_show