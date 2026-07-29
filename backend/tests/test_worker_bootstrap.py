from al_medlit.celery_app import app, celery_app, ping


def test_celery_app_declares_expected_queues():
    assert app is celery_app
    assert app.conf.task_default_queue == "control"
    assert {queue.name for queue in app.conf.task_queues} == {
        "control",
        "classical-cpu",
        "torch-cpu",
        "transformer-cpu",
        "peft-accelerator",
        "qlora-cuda",
    }


def test_celery_ping_task_runs_eagerly() -> None:
    previous_broker_url = app.conf.broker_url
    previous_task_always_eager = app.conf.task_always_eager

    app.conf.broker_url = "memory://"
    app.conf.task_always_eager = True
    try:
        assert ping.delay().get(timeout=1) == "pong"
    finally:
        app.conf.broker_url = previous_broker_url
        app.conf.task_always_eager = previous_task_always_eager
