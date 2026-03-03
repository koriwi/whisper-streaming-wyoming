"""Basic tests for wyoming-simul-whisper.

These tests validate the project structure, imports, and Wyoming protocol
integration without requiring a GPU or an actual Whisper model checkpoint.
"""

import asyncio
import struct
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

class TestImports(unittest.TestCase):
    def test_package_importable(self):
        import wyoming_simul_whisper
        self.assertIsNotNone(wyoming_simul_whisper.__version__)

    def test_simulstreaming_importable(self):
        """Vendored SimulStreaming config should be importable."""
        import dataclasses
        from simulstreaming.whisper.simul_whisper.config import AlignAttConfig
        field_names = {f.name for f in dataclasses.fields(AlignAttConfig)}
        self.assertIn("model_path", field_names)

    def test_token_buffer_importable(self):
        from simulstreaming.whisper.token_buffer import TokenBuffer
        self.assertTrue(callable(TokenBuffer.empty))

    def test_engine_module_importable(self):
        from wyoming_simul_whisper import engine
        self.assertTrue(hasattr(engine, "StreamingEngine"))

    def test_handler_module_importable(self):
        from wyoming_simul_whisper import handler
        self.assertTrue(hasattr(handler, "WhisperEventHandler"))

    def test_main_module_importable(self):
        from wyoming_simul_whisper import __main__
        self.assertTrue(callable(__main__.run))


# ---------------------------------------------------------------------------
# Engine unit tests (mocked model)
# ---------------------------------------------------------------------------

class TestStreamingEngineAudioConversion(unittest.TestCase):
    """Test PCM → float32 conversion without loading a real model."""

    def _make_engine_with_mock_model(self):
        """Return a StreamingEngine with its model replaced by a MagicMock."""
        import threading
        from wyoming_simul_whisper.engine import StreamingEngine

        mock_model = MagicMock()
        mock_model.tokenizer.decode.return_value = "hello"
        mock_model.tokens = [MagicMock()]
        mock_model.tokens[0].shape = (1, 4)
        mock_model.infer.return_value = ([1, 2, 3], {})
        mock_model.insert_audio.return_value = 0.0
        mock_model.segments = []

        # Bypass __init__ by creating the object and setting attributes directly
        engine = object.__new__(StreamingEngine)
        engine._lock = threading.Lock()
        engine._model = mock_model
        engine._audio_queue = []

        return engine

    def test_insert_audio_int16_conversion(self):
        import numpy as np
        engine = self._make_engine_with_mock_model()

        # 1 second of silence at 16 kHz, int16
        samples = np.zeros(16000, dtype=np.int16)
        pcm_bytes = samples.tobytes()

        engine.insert_audio(pcm_bytes, rate=16000, width=2, channels=1)

        self.assertEqual(len(engine._audio_queue), 1)
        chunk = engine._audio_queue[0]
        self.assertEqual(chunk.dtype, np.float32)
        self.assertEqual(len(chunk), 16000)
        # Silence should be all zeros
        self.assertTrue((chunk == 0.0).all())

    def test_insert_audio_stereo_downmix(self):
        import numpy as np
        engine = self._make_engine_with_mock_model()

        # Stereo: left = 10000, right = -10000  →  mean = 0
        left = np.full(160, 10000, dtype=np.int16)
        right = np.full(160, -10000, dtype=np.int16)
        interleaved = np.empty(320, dtype=np.int16)
        interleaved[0::2] = left
        interleaved[1::2] = right

        engine.insert_audio(interleaved.tobytes(), rate=16000, width=2, channels=2)

        chunk = engine._audio_queue[0]
        self.assertAlmostEqual(float(chunk.mean()), 0.0, places=5)

    def test_insert_audio_empty_noop(self):
        engine = self._make_engine_with_mock_model()
        engine.insert_audio(b"", rate=16000, width=2, channels=1)
        self.assertEqual(len(engine._audio_queue), 0)


# ---------------------------------------------------------------------------
# Handler protocol tests (fully mocked)
# ---------------------------------------------------------------------------

class TestHandlerProtocol(unittest.IsolatedAsyncioTestCase):
    """Verify that the handler sends the correct Wyoming events."""

    def _make_handler(self):
        from wyoming.info import AsrModel, AsrProgram, Attribution, Info
        from wyoming_simul_whisper.handler import WhisperEventHandler

        wyoming_info = Info(
            asr=[
                AsrProgram(
                    name="simul-whisper",
                    description="test",
                    attribution=Attribution(name="test", url="https://example.com"),
                    installed=True,
                    version="0.1.0",
                    models=[
                        AsrModel(
                            name="large-v3",
                            description="test model",
                            attribution=Attribution(name="test", url="https://example.com"),
                            installed=True,
                            version="0.1.0",
                            languages=["en"],
                        )
                    ],
                )
            ]
        )

        mock_engine = MagicMock()
        mock_engine.reset = MagicMock()
        mock_engine.insert_audio = MagicMock()
        mock_engine.process_iter = MagicMock(return_value="")
        mock_engine.finish = MagicMock(return_value="hello world")

        handler = object.__new__(WhisperEventHandler)
        handler._info = wyoming_info
        handler._engine = mock_engine
        handler._min_chunk_size = 0.01  # fast for tests
        handler._language = None
        handler._inference_task = None
        handler._partial_texts = []

        from wyoming.audio import AudioChunkConverter
        handler._converter = AudioChunkConverter(rate=16000, width=2, channels=1)

        sent_events = []

        async def _write_event(event):
            sent_events.append(event)

        handler.write_event = _write_event
        handler._sent_events = sent_events
        handler._mock_engine = mock_engine

        return handler

    async def test_describe_returns_info(self):
        from wyoming.info import Describe
        handler = self._make_handler()
        result = await handler.handle_event(Describe().event())
        self.assertTrue(result)
        self.assertEqual(len(handler._sent_events), 1)
        self.assertEqual(handler._sent_events[0].type, "info")

    async def test_full_transcription_cycle(self):
        import numpy as np
        from wyoming.asr import Transcribe
        from wyoming.audio import AudioChunk, AudioStart, AudioStop

        handler = self._make_handler()

        # Transcribe
        await handler.handle_event(Transcribe(language="en").event())

        # AudioStart
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await asyncio.sleep(0)  # yield to let task start

        # AudioChunk
        silence = np.zeros(1600, dtype=np.int16).tobytes()
        chunk = AudioChunk(rate=16000, width=2, channels=1, audio=silence)
        await handler.handle_event(chunk.event())

        # AudioStop → should call finish() and emit Transcript
        result = await handler.handle_event(AudioStop().event())
        self.assertFalse(result)  # session ends

        # Find the Transcript event
        transcript_events = [e for e in handler._sent_events if e.type == "transcript"]
        self.assertEqual(len(transcript_events), 1)

        from wyoming.asr import Transcript
        transcript = Transcript.from_event(transcript_events[0])
        self.assertEqual(transcript.text, "hello world")


# ---------------------------------------------------------------------------
# CLI argument parsing test
# ---------------------------------------------------------------------------

class TestCLI(unittest.TestCase):
    def test_arg_parsing(self):
        import argparse
        import sys
        from wyoming_simul_whisper.__main__ import run

        # Patch sys.argv and verify argparse parses without error
        test_args = [
            "wyoming-simul-whisper",
            "--uri", "tcp://0.0.0.0:10250",
            "--model", "large-v3",
            "--language", "en",
            "--min-chunk-size", "1.2",
        ]
        with patch("sys.argv", test_args), \
             patch("asyncio.run") as mock_run:
            run()
            mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
