from al_medlit.training.trainers.contracts import (
    TrainerPlugin,
    TrainerPluginRegistry,
    TrainerPreflight,
    TrainingInput,
    TrainingOutput,
    TrainingPlan,
    trainer_plugins,
)
from al_medlit.training.trainers.huggingface_sft import (
    HuggingFaceCausalSftTrainer,
    HuggingFaceSeq2SeqSftTrainer,
    register_huggingface_sft_trainers,
)
from al_medlit.training.trainers.huggingface_tasks import (
    HuggingFaceSequenceTrainer,
    HuggingFaceSpanTrainer,
    HuggingFaceTokenTrainer,
    register_huggingface_task_trainers,
)
from al_medlit.training.trainers.sklearn_tfidf import (
    SklearnTfidfTrainer,
    register_sklearn_tfidf_trainer,
)

register_sklearn_tfidf_trainer()
register_huggingface_task_trainers()
register_huggingface_sft_trainers()

__all__ = [
    "HuggingFaceCausalSftTrainer",
    "HuggingFaceSeq2SeqSftTrainer",
    "HuggingFaceSequenceTrainer",
    "HuggingFaceSpanTrainer",
    "HuggingFaceTokenTrainer",
    "SklearnTfidfTrainer",
    "TrainerPlugin",
    "TrainerPluginRegistry",
    "TrainerPreflight",
    "TrainingInput",
    "TrainingOutput",
    "TrainingPlan",
    "register_sklearn_tfidf_trainer",
    "register_huggingface_sft_trainers",
    "register_huggingface_task_trainers",
    "trainer_plugins",
]
