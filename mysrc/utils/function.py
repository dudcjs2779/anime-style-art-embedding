import torch


def gaussian_peak(
    sim,
    center=0.8,
    sigma=0.1,
    p=2.0,
    plateau=0.0,
    min_weight=0.2,
    max_weight=1.0,
):
    """
    Generalized Gaussian

    p = 2      : 정규분포
    p > 2      : center 주변은 완만해지고 중간 지점에 기울기 급격히 변함
    p < 2      : center의 뾰족한 정도

    plateau : center 주변의 평평한 구간
    """

    sim = torch.as_tensor(sim)

    d = (sim - center).abs()

    # Plateau
    d = torch.clamp(d - plateau, min=0.0)

    peak = torch.exp(-(d / sigma).pow(p))

    return min_weight + (max_weight - min_weight) * peak

def gaussian_valley(
    sim,
    center=0.8,
    sigma=0.1,
    p=2.0,
    plateau=0.0,
    min_weight=0.2,
    max_weight=2.0,
):
    """
    Generalized Gaussian Valley.
    center에 가까울 수록 낮은 값을 갖는 정규분포
    """

    sim = torch.as_tensor(sim)

    d = (sim - center).abs()

    d = torch.clamp(d - plateau, min=0.0)

    valley = 1.0 - torch.exp(-(d / sigma).pow(p))

    return min_weight + (max_weight - min_weight) * valley