"""
BL related function
"""


def implied_ret(lbd, cov, weight):
    """
    weight implied expected return
    """
    return lbd * cov.dot(weight)

