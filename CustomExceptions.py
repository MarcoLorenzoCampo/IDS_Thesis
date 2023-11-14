class accuracyException:
    """Raised when the accuracy of a model falls below a certain threshold."""
    pass

class precisionException:
    """Raised when the precision of a model falls below a certain threshold."""
    pass

class f1Exception:
    """Raised when the F1 score of a model falls below a certain threshold."""
    pass

class tprException:
    """Raised when the true positive rate (TPR) of a model falls below a certain threshold."""
    pass

class fprException:
    """Raised when the false positive rate (FPR) of a model rises above a certain threshold."""
    pass

class tnrException:
    """Raised when the true negative rate (TNR) of a model falls below a certain threshold."""
    pass

class fnrException:
    """Raised when the false negative rate (FNR) of a model rises above a certain threshold."""
    pass

class quarantineRatioException:
    """Raised when the ratio of quarantined data points to total data points falls below a certain threshold."""
    pass

