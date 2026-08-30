class APIError(Exception):
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(APIError):
    status_code = 404


class ValidationError(APIError):
    status_code = 422


class ConflictError(APIError):
    status_code = 409


class ForbiddenError(APIError):
    status_code = 403


class UnauthorizedError(APIError):
    status_code = 401


class RateLimitedError(APIError):
    status_code = 429
