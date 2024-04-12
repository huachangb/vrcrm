from ..poem.Skylines import PRMWrapper

class PRMWrapperBackwardSupport(PRMWrapper):
    def __call__(self, X):
        return self.labeler.predict_proba(X)
