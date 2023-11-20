class accuracyException(Exception):
    """Raised when the accuracy of a model falls below a certain threshold."""
    pass

class precisionException(Exception):
    """Raised when the precision of a model falls below a certain threshold."""
    pass

class f1Exception(Exception):
    """Raised when the F1 score of a model falls below a certain threshold."""
    pass

class tprException(Exception):
    """Raised when the true positive rate (TPR) of a model falls below a certain threshold."""
    pass

class fprException(Exception):
    """Raised when the false positive rate (FPR) of a model rises above a certain threshold."""
    pass

class tnrException(Exception):
    """Raised when the true negative rate (TNR) of a model falls below a certain threshold."""
    pass

class fnrException(Exception):
    """Raised when the false negative rate (FNR) of a model rises above a certain threshold."""
    pass

class quarantineRatioException(Exception):
    """Raised when the ratio of quarantined data points to total data points falls below a certain threshold."""
    pass

