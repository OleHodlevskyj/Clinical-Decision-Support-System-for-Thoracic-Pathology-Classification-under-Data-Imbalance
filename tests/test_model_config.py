#побудова архітектур і узгодженість конфігурації класів
import pytest
import torch

from config import CLASS_NAMES, UA_LABELS, K_CRITICALITY
from model import build_model


@pytest.mark.parametrize('arch', ['resnet50', 'efficientnet_b0'])
def test_model_outputs_one_logit_per_class(arch):
    #обидві архітектури мають давати для пакета з 2 знімків рівно 14 виходів
    #на кожен знімок, по одному на патологію
    m = build_model(pretrained=False, arch=arch).eval()
    with torch.no_grad():
        assert m(torch.zeros(2, 3, 224, 224)).shape == (2, len(CLASS_NAMES))


def test_architecture_from_environment(monkeypatch):
    #архітектуру можна вибрати змінною середовища ARCH, як під час навчання
    monkeypatch.setenv('ARCH', 'efficientnet_b0')
    assert type(build_model(pretrained=False)).__name__ == 'EfficientNet'


def test_unknown_architecture_is_rejected():
    #непідтримувана архітектура має зупиняти програму зрозумілим повідомленням
    with pytest.raises(SystemExit):
        build_model(pretrained=False, arch='vgg16')


def test_every_class_has_label_and_criticality():
    #узгодженість config.py: 14 різних патологій, у кожної є український підпис
    #і коефіцієнт k_c у межах від 1 до 3
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES)) == 14
    assert set(UA_LABELS) == set(K_CRITICALITY) == set(CLASS_NAMES)
    assert all(1.0 <= k <= 3.0 for k in K_CRITICALITY.values())
