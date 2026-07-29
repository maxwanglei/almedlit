from al_medlit.training.recipe_contracts import Parameterization, RuntimeClass, TaskKind
from al_medlit.training.recipe_registry import training_recipes


def test_recipe_catalog_spans_linear_models_transformers_and_llms():
    recipes = {recipe.key: recipe for recipe in training_recipes.list()}

    assert {
        "tfidf_linear_regression",
        "tfidf_logistic_regression",
        "transformer_sequence_classification",
        "transformer_token_classification",
        "transformer_span_extraction",
        "causal_lm_sft_full",
        "causal_lm_sft_lora",
        "causal_lm_sft_qlora",
        "seq2seq_lm_sft_full",
        "seq2seq_lm_sft_lora",
        "seq2seq_lm_sft_qlora",
    } <= recipes.keys()
    assert recipes["tfidf_linear_regression"].supported_task_kinds == (
        TaskKind.REGRESSION,
    )
    assert (
        recipes["causal_lm_sft_qlora"].parameterization
        == Parameterization.QLORA
    )
    assert (
        recipes["causal_lm_sft_qlora"].environment.runtime_class
        == RuntimeClass.QLORA_CUDA
    )
    assert recipes["tfidf_linear_regression"].environment.packages == (
        "scikit-learn",
        "skops",
    )


def test_recipe_validation_normalizes_and_rejects_invalid_config():
    valid = training_recipes.validate(
        "tfidf_linear_regression",
        {
            "fields": {"input_field": "text", "target_field": "score"},
            "ngram_min": 1,
            "ngram_max": 3,
        },
    )
    invalid = training_recipes.validate(
        "tfidf_linear_regression",
        {
            "fields": {"input_field": "text", "target_field": "score"},
            "ngram_min": 4,
            "ngram_max": 2,
        },
    )

    assert valid.valid is True
    assert valid.normalized_config is not None
    assert valid.normalized_config["max_features"] == 50_000
    assert invalid.valid is False
    assert any("ngram_max" in message for message in invalid.errors)


def test_recipe_catalog_does_not_require_optional_packages_in_api_process():
    for recipe in training_recipes.list():
        assert recipe.environment.requires_verified_environment is True
        assert recipe.environment.setup_hint
        assert recipe.config_schema["type"] == "object"


def test_sft_templates_reject_attribute_access_and_unversioned_shapes():
    invalid = training_recipes.validate(
        "causal_lm_sft_lora",
        {
            "base_model_asset_id": 1,
            "prompt_field": "prompt",
            "completion_field": "answer",
            "prompt_template_version": "v1",
            "input_template": "{prompt.__class__}",
            "completion_template": "{completion}",
        },
    )

    assert invalid.valid is False
    assert any("input_template" in message for message in invalid.errors)
