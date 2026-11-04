import pytest
import torch
from torch import nn

from src.models.factory import create_model


@pytest.mark.parametrize("model_name", ["rnn", "lstm"])
def test_model_forward_shape_and_no_nan(model_name):
    model = create_model(model_name, input_size=15)
    x = torch.randn(8, 24, 15)
    out = model(x)
    assert out.shape == (8, 1)
    assert not torch.isnan(out).any()


def test_create_model_invalid_name():
    with pytest.raises(ValueError, match="Modelo desconocido"):
        create_model("xyz", input_size=15)


@pytest.mark.parametrize("model_name", ["rnn", "lstm"])
def test_minimal_training_convergence_and_gradients(model_name):
    torch.manual_seed(42)
    model = create_model(model_name, input_size=15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    torch.manual_seed(123)
    x = torch.randn(16, 24, 15)
    y = torch.randn(16, 1)

    initial_loss = None
    first_step_grads_ok = False

    for step in range(40):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()

        if step == 0:
            first_step_grads_ok = all(
                p.grad is not None for p in model.parameters() if p.requires_grad
            )

        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss
    assert first_step_grads_ok
