from al_medlit.core.exceptions import ValidationError
from al_medlit.training.contracts import TaskContractDescriptor, TaskEvaluator


class TaskEvaluatorRegistry:
    def __init__(self) -> None:
        self._evaluators: dict[str, TaskEvaluator] = {}

    def register(self, evaluator: TaskEvaluator, *, replace: bool = False) -> None:
        key = evaluator.descriptor.key
        if key in self._evaluators and not replace:
            raise ValidationError(f"Task evaluator '{key}' is already registered")
        self._evaluators[key] = evaluator

    def get(self, key: str) -> TaskEvaluator:
        try:
            return self._evaluators[key]
        except KeyError as exc:
            raise ValidationError(f"Unknown task evaluator '{key}'") from exc

    def list(self) -> tuple[TaskEvaluator, ...]:
        return tuple(self._evaluators[key] for key in sorted(self._evaluators))

    def list_descriptors(self) -> tuple[TaskContractDescriptor, ...]:
        return tuple(evaluator.descriptor for evaluator in self.list())


task_evaluators = TaskEvaluatorRegistry()
