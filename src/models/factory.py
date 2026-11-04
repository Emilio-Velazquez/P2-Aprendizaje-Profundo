from torch import nn

from src.models.lstm import LSTMForecaster
from src.models.rnn import RNNForecaster


def create_model(
    name: str,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    output_size: int = 1,
) -> nn.Module:
    normalized_name = name.lower().strip()
    if normalized_name == "rnn":
        return RNNForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
        )
    if normalized_name == "lstm":
        return LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
        )
    raise ValueError(
        f"Modelo desconocido '{name}'. Las opciones validas son 'rnn' y 'lstm'."
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
