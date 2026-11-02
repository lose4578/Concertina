import importlib.util
from pathlib import Path

# The tiny-backbone fixtures live with the model tests; load that conftest as a plain module and re-export them.
_spec = importlib.util.spec_from_file_location(
    "concertina_model_conftest", Path(__file__).parents[1] / "model" / "conftest.py"
)
_model_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model_conftest)

tiny_backbone = _model_conftest.tiny_backbone
tiny_config = _model_conftest.tiny_config
make_batch = _model_conftest.make_batch
