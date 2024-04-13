import torch
import numpy as np

from ..poem.Skylines import PRMWrapper

class PRMWrapperBackwardSupport(PRMWrapper):
    def __call__(self, X):
        X_np = X.numpy()
        probs =  self.labeler.predict_proba(X_np).astype(np.float32)
        return torch.from_numpy(probs)
