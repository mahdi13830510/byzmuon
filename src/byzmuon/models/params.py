from torch import nn


def muon_excluded(model: nn.Module) -> set[str]:
    """Matrix parameters Muon should leave to the fallback optimizer: embeddings and the output head."""
    excluded: set[str] = set()
    linears: list[str] = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            excluded.add(f"{mod_name}.weight")
        elif isinstance(mod, nn.Linear):
            linears.append(mod_name)
    if linears:  # the last Linear is the classification head
        excluded.add(f"{linears[-1]}.weight")
    return excluded


def trainable_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]
