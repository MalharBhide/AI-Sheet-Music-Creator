import io

import pytest
import torch
from left_precision_v40 import CONTEXT_WIDTH
from left_precision_v48 import model


def inputs():
    generator = torch.Generator().manual_seed(918273)
    return [torch.rand(shape, generator=generator) for shape in
            ((3, 40, 9), (3, 61, 18), (3, 61, 18), (3, CONTEXT_WIDTH))]


@pytest.mark.parametrize("width", [12, 20])
def test_new_model_starts_with_conservative_permissions_and_roundtrips(width):
    torch.manual_seed(1937)
    network = model(width).eval()
    values = inputs()
    with torch.no_grad():
        note, frame = network(*values)
    assert note.shape == (3, 2) and frame.shape == (3, 40)
    assert torch.all(torch.softmax(note, dim=1)[:, 0] > .98)
    assert torch.all(torch.sigmoid(frame) > .98)
    stream = io.BytesIO()
    torch.save(network.state_dict(), stream)
    stream.seek(0)
    loaded = model(width).eval()
    loaded.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    with torch.no_grad():
        for expected, actual in zip((note, frame), loaded(*values), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_frame_support_can_learn_from_both_raw_boundary_windows():
    torch.manual_seed(1937)
    network = model(12).eval()
    # Open the deliberately conservative final head to test the learned paths.
    torch.nn.init.constant_(network.frame.weight, .1)
    values = [value.requires_grad_() for value in inputs()]
    _, frame = network(*values)
    frame.sum().backward()
    for value in values:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.any(value.grad != 0)
    with torch.no_grad():
        for index in (1, 2):
            changed = list(values)
            changed[index] = torch.zeros_like(changed[index])
            _, alternate = network(*changed)
            assert not torch.equal(alternate, frame)


def test_unsupported_encoder_width_fails_before_training():
    with pytest.raises(ValueError):
        model(16)
