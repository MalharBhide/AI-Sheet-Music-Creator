import io

import pytest
import torch
from left_precision_v40 import CONTEXT_WIDTH
from left_precision_v58 import frame_phases, model, require_phase_contract


def inputs(count=3):
    generator = torch.Generator().manual_seed(137319)
    return [torch.rand(shape, generator=generator) for shape in
            ((count, 40, 9), (count, 61, 18), (count, 61, 18), (count, CONTEXT_WIDTH))]


def test_phase_clock_matches_existing_candidate_interior_without_inventing_seconds():
    phase = frame_phases()
    assert phase.shape == (1, 4, 40)
    assert phase.dtype == torch.float32 and not phase.requires_grad
    assert torch.equal(phase[0, 1:].sum(dim=0), torch.ones(40))
    assert torch.equal(phase[0, 1].nonzero().ravel(), torch.arange(8))
    assert torch.equal(phase[0, 2].nonzero().ravel(), torch.arange(8, 32))
    assert torch.equal(phase[0, 3].nonzero().ravel(), torch.arange(32, 40))
    assert phase[0, 0, 0] == -1 and phase[0, 0, -1] == 1
    assert torch.all(torch.diff(phase[0, 0]) > 0)


@pytest.mark.parametrize("width", [12, 20])
def test_fresh_model_is_conservative_and_checkpoint_roundtrip_is_exact(width):
    torch.manual_seed(373193)
    network = model(width).eval()
    values = inputs()
    require_phase_contract(network)
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
    require_phase_contract(loaded)
    with torch.no_grad():
        for actual, expected in zip(loaded(*values), (note, frame), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["rank", "interior", "dtype", "gradient"])
def test_checkpoint_cannot_silently_change_the_candidate_phase_clock(mutation):
    network = model(12)
    if mutation == "rank":
        network.phase[0, 0, 0] = 0
    elif mutation == "interior":
        network.phase[0, 2, 8] = 0
    elif mutation == "dtype":
        network.phase = network.phase.double()
    else:
        network.phase.requires_grad_(True)
    with pytest.raises(ValueError, match="phase clock"):
        require_phase_contract(network)


def test_frame_predictions_can_use_every_physical_input_and_fixed_phase_channel():
    torch.manual_seed(373193)
    network = model(12).eval()
    torch.nn.init.constant_(network.frame.weight, .1)
    values = [value.requires_grad_() for value in inputs()]
    _, frame = network(*values)
    frame.sum().backward()
    for value in values:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    first = network.presence[0].weight.grad
    assert first is not None and torch.isfinite(first).all()
    assert all(torch.count_nonzero(first[:, index]) > 0 for index in range(9, 13))
    assert network.phase.grad is None


@pytest.mark.parametrize("index", range(4))
def test_wrong_frame_clock_is_rejected_before_prediction(index):
    values = inputs()
    values[index] = values[index][:, :-1]
    with pytest.raises(ValueError, match="observation clock"):
        model(12)(*values)


def test_validated_batching_does_not_change_frame_or_note_predictions():
    torch.manual_seed(373193)
    network = model(12).eval()
    torch.nn.init.constant_(network.frame.weight, .1)
    values = inputs(4)
    with torch.inference_mode():
        full = network(*values)
        parts = [network(*[value[index:index+1] for value in values]) for index in range(4)]
    for index in range(2):
        torch.testing.assert_close(full[index], torch.cat([part[index] for part in parts]), rtol=1e-5, atol=1e-6)


def test_initial_contract_proves_all_fitting_permissions_are_keep():
    from train_left_precision_v58 import initial_contract

    receipt = initial_contract(model(12), inputs())
    assert receipt["fitting_candidates"] == 3
    assert receipt["all_additional_permissions_keep"] is True
    assert receipt["no_validation_or_test_read"] is True


def test_initial_contract_catches_accidentally_inherited_rejecting_weights():
    from train_left_precision_v58 import initial_contract

    network = model(12)
    with torch.no_grad():
        network.frame.bias.fill_(-10)
        network.reject.classifier[-1].bias.copy_(torch.tensor([0., 10.]))
    with pytest.raises(ValueError, match="initial permissions"):
        initial_contract(network, inputs())


def test_initial_contract_requires_a_complete_nonempty_fitting_population():
    from train_left_precision_v58 import initial_contract

    with pytest.raises(ValueError, match="Incomplete fitting inputs"):
        initial_contract(model(12), inputs(0))
