"""Source-independent W14 projection; probability arrays are supplied by callers."""
try:
    from .projection import EIoUMixin
except ImportError:
    from w14_projection import EIoUMixin


def project_refiner(probabilities, matrices, scales, seeds, scores, projection_class,
                    config=None, eiou_binarize=True):
    """Accept aligned refiner arrays from either OOF or test, with W6 fallbacks."""
    assert len(probabilities) == len(matrices) == len(scales) == len(seeds) == len(scores)
    class EIoUProjection(EIoUMixin, projection_class):
        pass
    cls = EIoUProjection if eiou_binarize else projection_class
    projection = cls(None, None, seeds, probabilities, matrices, scales)
    rule = dict(config or {'threshold': .45, 'owner': 'original'})
    if eiou_binarize:
        rule['threshold'] = 'eiou'
    return projection.apply(rule, scores)
