"""pneumonia_cnn: CNN architectures for pediatric chest X-ray classification."""

from .config import TrainingConfig
from .models import MODEL_REGISTRY, build_model

__all__ = ["TrainingConfig", "MODEL_REGISTRY", "build_model"]
__version__ = "1.0.0"
