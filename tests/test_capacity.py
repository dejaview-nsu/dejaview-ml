import asyncio
import threading
from unittest.mock import AsyncMock

import httpx
import pytest
from test_api import TestEncoders, png
from test_text_audio import wav_bytes

from dejaview_ml.app import create_app
from dejaview_ml.capacity import DECODING, Capacity
from dejaview_ml.config import Settings
from dejaview_ml.errors import MlError


class Gate:
    """A model run that holds its model until the test opens the gate."""

    def __init__(self):
        self.entered = threading.Event()
        self.opened = threading.Event()

    def __call__(self, result=None):
        self.entered.set()
        assert self.opened.wait(5)
        return result

    async def wait_entered(self):
        assert await asyncio.to_thread(self.entered.wait, 5)


async def test_query_waits_for_its_model_instead_of_being_rejected():
    capacity, gate = Capacity(slots=2, queue_limit=4, wait_seconds=5), Gate()
    first = asyncio.create_task(capacity.run("image", capacity.budget(), gate, 1))
    await gate.wait_entered()
    second = asyncio.create_task(capacity.run("image", capacity.budget(), lambda: 2))
    await asyncio.sleep(0.05)
    assert not second.done()  # the image model is busy: the second query waits
    gate.opened.set()
    assert await first == 1
    assert await second == 2


async def test_other_models_run_while_one_is_busy():
    capacity, gate = Capacity(slots=2, queue_limit=4, wait_seconds=0), Gate()
    held = asyncio.create_task(capacity.run("audio", capacity.budget(), gate))
    await gate.wait_entered()
    try:
        # Zero wait budget: text must not wait at all for the busy audio model.
        assert await capacity.run("text", capacity.budget(), lambda: "text") == "text"
    finally:
        gate.opened.set()
    await held


async def test_cpu_slots_limit_models_running_at_once():
    capacity, gate = Capacity(slots=1, queue_limit=4, wait_seconds=0.1), Gate()
    held = asyncio.create_task(capacity.run("audio", capacity.budget(), gate))
    await gate.wait_entered()
    try:
        with pytest.raises(MlError) as error:
            await capacity.run("text", capacity.budget(), lambda: "text")
        assert (error.value.status, error.value.code) == (503, "NOT_READY")
    finally:
        gate.opened.set()
    await held


async def test_wait_budget_is_shared_by_the_steps_of_one_request():
    capacity, gate = Capacity(slots=2, queue_limit=4, wait_seconds=0.2), Gate()
    held = asyncio.create_task(capacity.run("speech", capacity.budget(), gate))
    await gate.wait_entered()
    budget = capacity.budget()
    try:
        await capacity.run(DECODING, budget, lambda: None)
        with pytest.raises(MlError):
            await capacity.run("speech", budget, lambda: None)
        assert budget.seconds <= 0.01
        # Nothing left to wait with, but a free model is still taken at once.
        assert await capacity.run("text", budget, lambda: "text") == "text"
    finally:
        gate.opened.set()
    await held


async def test_full_queue_is_rejected_at_once():
    capacity, gate = Capacity(slots=2, queue_limit=1, wait_seconds=5), Gate()
    held = asyncio.create_task(capacity.run("image", capacity.budget(), gate))
    await gate.wait_entered()
    waiting = asyncio.create_task(capacity.run("image", capacity.budget(), lambda: 2))
    await asyncio.sleep(0.05)
    try:
        with pytest.raises(MlError):
            await asyncio.wait_for(capacity.run("image", capacity.budget(), lambda: 3), 1)
    finally:
        gate.opened.set()
    await held
    assert await waiting == 2


async def test_cancelled_request_keeps_the_model_until_its_thread_is_done():
    capacity, gate = Capacity(slots=2, queue_limit=4, wait_seconds=5), Gate()
    gone = asyncio.create_task(capacity.run("image", capacity.budget(), gate))
    await gate.wait_entered()
    gone.cancel()  # the client disconnected; the thread still uses the model
    with pytest.raises(asyncio.CancelledError):
        await gone
    after = asyncio.create_task(capacity.run("image", capacity.budget(), lambda: "next"))
    await asyncio.sleep(0.05)
    assert not after.done()
    gate.opened.set()
    assert await after == "next"


async def test_text_search_is_served_while_audio_transcribes():
    gate = Gate()

    class SlowSpeech(TestEncoders):
        def _transcribe(self, waveform):
            return gate("иди за белым кроликом")

    store = AsyncMock()
    store.healthy.return_value = True
    store.search.return_value = [603]
    store.search_audio.return_value = [603]
    app = create_app(Settings(_env_file=None), SlowSpeech(), store)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://ml"
        ) as c:
            audio = asyncio.create_task(
                c.post(
                    "/v1/search/audio", content=wav_bytes(), headers={"Content-Type": "audio/wav"}
                )
            )
            await gate.wait_entered()
            try:
                text = await c.post("/v1/search/text", json={"query": "Матрица"})
                assert text.status_code == 200
                image = await c.post(
                    "/v1/search/image", content=png(), headers={"Content-Type": "image/png"}
                )
                assert image.status_code == 200
            finally:
                gate.opened.set()
            assert (await audio).status_code == 200
    assert store.search_audio.call_args.args[0].transcript == "иди за белым кроликом"
