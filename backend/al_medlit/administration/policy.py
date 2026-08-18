from dataclasses import dataclass

from sqlalchemy.orm import Session

from al_medlit.administration.models import InstancePolicy
from al_medlit.core.config import settings

DEFAULT_INVITE_EXPIRY_MINUTES = 10_080
DEFAULT_ACCOUNT_ACTION_EXPIRY_MINUTES = 60


@dataclass(frozen=True)
class EffectiveInstancePolicy:
    allow_self_registration: bool
    default_invite_expiry_minutes: int
    account_action_expiry_minutes: int


def get_effective_policy(db: Session) -> EffectiveInstancePolicy:
    """Read policy without mutating the transaction.

    Deployments upgraded from an earlier release continue honoring their
    environment-level self-registration value until an administrator stores a
    database override.
    """

    policy = db.get(InstancePolicy, 1)
    return EffectiveInstancePolicy(
        allow_self_registration=(
            settings.allow_self_registration
            if policy is None or policy.allow_self_registration is None
            else policy.allow_self_registration
        ),
        default_invite_expiry_minutes=(
            DEFAULT_INVITE_EXPIRY_MINUTES
            if policy is None
            else policy.default_invite_expiry_minutes
        ),
        account_action_expiry_minutes=(
            DEFAULT_ACCOUNT_ACTION_EXPIRY_MINUTES
            if policy is None
            else policy.account_action_expiry_minutes
        ),
    )
