def test_registry_has_eight_models():
    from src.models import MODEL_REGISTRY
    assert len(MODEL_REGISTRY) == 8
