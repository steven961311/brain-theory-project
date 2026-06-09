import numpy as np

from etam_mnist.templates import decode_template, digit_templates


def test_digit_templates_are_bipolar_unique_and_decodable():
    templates = digit_templates()
    assert templates.shape == (10, 96)
    assert templates.dtype == np.int8
    assert set(np.unique(templates)) == {-1, 1}
    assert np.unique(templates, axis=0).shape[0] == 10
    for label, template in enumerate(templates):
        assert decode_template(template) == label


def test_template_tie_break_is_deterministic():
    assert decode_template(np.zeros(96, dtype=np.int8)) == int(
        np.argmax(digit_templates().sum(axis=1) * 0)
    )

