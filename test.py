"""Silent Guard: real-time, silent-by-default hazard awareness assistant (All-in-One)."""
from __future__ import annotations

import argparse
import base64
import collections
from collections import Counter, deque
import heapq
import itertools
import logging
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum, auto
from logging.handlers import RotatingFileHandler
from typing import Callable, Deque, Dict, List, Optional, Protocol, Tuple, Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_CLASS_HEIGHTS_M: Dict[int, float] = {
    0: 1.70,  # person
    1: 1.10,  # bicycle
    2: 1.50,  # car
    3: 1.20,  # motorcycle
    5: 3.20,  # bus
    7: 3.00,  # truck
}

@dataclass(frozen=True)
class CameraConfig:
    source: Union[int, str] = "street_video.mp4"
    frame_size: Tuple[int, int] = (960, 540)
    horizontal_fov_deg: float = 70.0
    mount_height_m: float = 1.30
    horizon_ratio: float = 0.50
    open_retries: int = 3
    reconnect_backoff_s: float = 2.0
    max_consecutive_read_failures: int = 30

@dataclass(frozen=True)
class DetectorConfig:
    weights: str = "yolo11n.pt"  
    tracker: str = "bytetrack.yaml"
    classes: Tuple[int, ...] = (0, 1, 2, 3, 5, 7)
    conf: float = 0.25
    imgsz: int = 320
    device: str = "cpu"  

@dataclass(frozen=True)
class HealthConfig:
    brightness_threshold: float = 20.0
    variance_threshold: float = 15.0
    fail_frames_to_trip: int = 8
    ok_frames_to_clear: int = 5

@dataclass(frozen=True)
class ProximityConfig:
    class_heights_m: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_CLASS_HEIGHTS_M))
    window_s: float = 0.8
    min_samples: int = 5
    min_track_age_s: float = 0.4
    track_timeout_s: float = 1.0
    ema_alpha: float = 0.5
    min_expansion_tstat: float = 2.0
    ground_pitch_err_px: float = 15.0
    critical_ttc_s: float = 2.5
    critical_distance_m: float = 20.0
    min_box_height_px: float = 24.0
    immediate_distance_m: float = 1.2
    corridor_half_width_m: float = 0.9
    approach_ttc_s: float = 8.0
    enter_hits: int = 3
    enter_window: int = 4
    exit_misses: int = 6
    emergency_min_conf: float = 0.50

@dataclass(frozen=True)
class AlertConfig:
    track_cooldown_s: float = 2.5
    zone_cooldown_s: float = 4.0
    emergency_ttl_s: float = 1.5
    summary_ttl_s: float = 6.0

@dataclass(frozen=True)
class AudioConfig:
    enable_tts: bool = True
    enable_mic: bool = True
    tts_rate: int = 5
    tts_startup_timeout_s: float = 10.0
    tts_max_restarts: int = 5
    mic_phrase_limit_s: float = 3.0
    mic_listen_timeout_s: float = 1.0
    mic_recognize_timeout_s: float = 5.0
    wake_phrases: Tuple[str, ...] = (
        "status", "what's around", "what is around", "around me", "surroundings",
    )

@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    proximity: ProximityConfig = field(default_factory=ProximityConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    headless: bool = False
    window_name: str = "Vision Feed"
    log_level: str = "INFO"
    log_file: str = "silent_guard.log"

# ---------------------------------------------------------------------------
# 2. LOGGING SETUP
# ---------------------------------------------------------------------------

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-8s [%(threadName)-12s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, _DATEFMT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        try:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            root.warning("File logging disabled (%s): %s", log_file, exc)

    for noisy in ("ultralytics", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

class RateLimitedLog:
    def __init__(self, logger: logging.Logger, interval_s: float = 10.0) -> None:
        self._logger = logger
        self._interval = interval_s
        self._last: Dict[str, float] = {}
        self._suppressed: Dict[str, int] = {}
        self._lock = threading.Lock()

    def log(self, level: int, key: str, msg: str, *args: object) -> None:
        now = time.monotonic()
        with self._lock:
            last = self._last.get(key)
            if last is not None and now - last < self._interval:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                return
            suppressed = self._suppressed.pop(key, 0)
            self._last[key] = now
        if suppressed:
            msg = f"{msg} (+{suppressed} similar suppressed)"
        self._logger.log(level, msg, *args)

log = logging.getLogger("silent_guard")

# ---------------------------------------------------------------------------
# 3. TEXT-TO-SPEECH (TTS) SERVICE
# ---------------------------------------------------------------------------

DoneCallback = Callable[[int], None]
DeadCallback = Callable[[str], None]

class Priority(IntEnum):
    EMERGENCY = 0
    SUMMARY = 1
    INFO = 2

class TTSUnavailable(RuntimeError):
    def __init__(self, msg: str, permanent: bool = False) -> None:
        super().__init__(msg)
        self.permanent = permanent

@dataclass(order=True)
class _Utterance:
    priority: int
    seq: int
    text: str = field(compare=False)
    expires_at: float = field(compare=False)

class TTSBackend(Protocol):
    name: str
    def start(self, on_done: DoneCallback, on_dead: DeadCallback) -> None: ...
    def speak(self, uid: int, text: str) -> None: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...

class LogOnlyBackend:
    name = "log-only"
    def __init__(self) -> None:
        self._on_done: Optional[DoneCallback] = None

    def start(self, on_done: DoneCallback, on_dead: DeadCallback) -> None:
        self._on_done = on_done

    def speak(self, uid: int, text: str) -> None:
        if self._on_done:
            self._on_done(uid)

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass

_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
function Emit([string]$m) { [Console]::Out.WriteLine($m); [Console]::Out.Flush() }
try {
    Add-Type -AssemblyName System.Speech
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.Rate = _RATE_
    if (_MUTE_) { $s.SetOutputToNull() } else { $s.SetOutputToDefaultAudioDevice() }
} catch {
    Emit ('FATAL ' + $_.Exception.Message); exit 1
}
$in = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), (New-Object System.Text.UTF8Encoding($false)))
Emit 'READY'
$pending = $in.ReadLineAsync()
$prompt = $null
$promptId = ''
while ($true) {
    if ($pending.Wait(20)) {
        $line = $pending.Result
        if ($line -eq $null -or $line -eq 'QUIT') { break }
        if ($line -eq 'CANCEL') {
            $s.SpeakAsyncCancelAll()
        } elseif ($line.StartsWith('SAY ')) {
            $rest = $line.Substring(4)
            $sep = $rest.IndexOf(' ')
            if ($sep -gt 0) {
                $promptId = $rest.Substring(0, $sep)
                try { $prompt = $s.SpeakAsync($rest.Substring($sep + 1)) }
                catch { Emit ('ERR ' + $_.Exception.Message); Emit ('DONE ' + $promptId); $prompt = $null }
            }
        }
        $pending = $in.ReadLineAsync()
    }
    if ($prompt -ne $null -and $prompt.IsCompleted) {
        Emit ('DONE ' + $promptId)
        $prompt = $null
    }
}
try { $s.SpeakAsyncCancelAll(); $s.Dispose() } catch { }
"""

class PowerShellBackend:
    name = "windows-sapi"
    def __init__(self, rate: int = 1, startup_timeout_s: float = 10.0, mute: bool = False) -> None:
        self._rate = max(-10, min(10, int(rate)))
        self._startup_timeout = startup_timeout_s
        self._mute = mute
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()
        self._ready = threading.Event()
        self._alive = False
        self._started = False
        self._closing = False
        self._fatal: Optional[str] = None
        self._on_done: Optional[DoneCallback] = None
        self._on_dead: Optional[DeadCallback] = None

    def start(self, on_done: DoneCallback, on_dead: DeadCallback) -> None:
        if sys.platform != "win32":
            raise TTSUnavailable("PowerShell SAPI backend requires Windows", permanent=True)
        self._on_done, self._on_dead = on_done, on_dead
        script = (_PS_SCRIPT.replace("_RATE_", str(self._rate))
                            .replace("_MUTE_", "$true" if self._mute else "$false"))
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NoLogo", "-NoProfile", "-NonInteractive",
                 "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise TTSUnavailable(f"cannot launch powershell: {exc}", permanent=True) from exc

        self._alive = True
        self._reader = threading.Thread(target=self._read_loop, name="tts-reader", daemon=True)
        self._reader.start()

        if not self._ready.wait(self._startup_timeout) or not self._alive:
            fatal = self._fatal
            self.close()
            if fatal:
                raise TTSUnavailable(f"speech engine error: {fatal}", permanent=True)
            raise TTSUnavailable("speech engine did not become ready in time")
        self._started = True

    def speak(self, uid: int, text: str) -> None:
        self._write(f"SAY {uid} {text}")

    def cancel(self) -> None:
        self._write("CANCEL")

    def close(self) -> None:
        self._closing = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                self._write("QUIT")
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=2.0)
        self._alive = False

    def _write(self, line: str) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or not self._alive:
            raise BrokenPipeError("speech engine is not running")
        with self._write_lock:
            proc.stdin.write((line + "\n").encode("utf-8"))
            proc.stdin.flush()

    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if line == "READY":
                    self._ready.set()
                elif line.startswith("DONE "):
                    try:
                        uid = int(line[5:])
                    except ValueError:
                        continue
                    if self._on_done:
                        self._on_done(uid)
                elif line.startswith("FATAL "):
                    self._fatal = line[6:]
                    log.error("Speech engine: %s", self._fatal)
                elif line.startswith("ERR "):
                    log.warning("Speech engine: %s", line[4:])
        except (OSError, ValueError) as exc:
            log.debug("Speech engine reader stopped: %s", exc)
        finally:
            self._alive = False
            self._ready.set()
            if self._started and not self._closing and self._on_dead:
                self._on_dead(self._fatal or f"process exited (code {proc.poll()})")

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]+")

def _sanitize(text: str, max_len: int = 300) -> str:
    return " ".join(_CONTROL_CHARS.sub(" ", text).split())[:max_len]

def _supersedes(new: int, queued: int) -> bool:
    return queued > new or (new == queued == Priority.SUMMARY)

class TTSService:
    def __init__(
        self,
        cfg: AudioConfig,
        backend_factory: Optional[Callable[[], TTSBackend]] = None,
        max_queue: int = 8,
    ) -> None:
        self._cfg = cfg
        self._factory = backend_factory or self._default_factory
        self._max_queue = max_queue
        self._cond = threading.Condition()
        self._heap: List[_Utterance] = []
        self._seq = itertools.count(1)
        self._current: Optional[_Utterance] = None
        self._current_done = False
        self._current_deadline = 0.0
        self._preempt = False
        self._backend_dead: Optional[str] = None
        self._stopping = False
        self._drain_deadline = 0.0
        self._stop_evt = threading.Event()
        self._backend: TTSBackend = LogOnlyBackend()
        self._backend_state = "starting"
        self._restart_times: Deque[float] = collections.deque()
        self._thread = threading.Thread(target=self._run, name="tts-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def say(self, text: str, priority: Priority = Priority.INFO, ttl_s: float = 5.0) -> bool:
        clean = _sanitize(text)
        if not clean:
            return False
        now = time.monotonic()
        with self._cond:
            if self._stopping:
                return False
            if self._current is not None and self._current.text == clean:
                return False
            if any(u.text == clean for u in self._heap):
                return False
            kept = [u for u in self._heap if not _supersedes(priority, u.priority)]
            if len(kept) != len(self._heap):
                self._heap = kept
                heapq.heapify(self._heap)
            utt = _Utterance(int(priority), next(self._seq), clean, now + ttl_s)
            heapq.heappush(self._heap, utt)
            while len(self._heap) > self._max_queue:
                self._heap.remove(max(self._heap))
                heapq.heapify(self._heap)
            if self._current is not None and priority < self._current.priority:
                self._preempt = True
            self._cond.notify_all()
            return utt in self._heap

    def stop(self, drain_timeout_s: float = 3.0) -> None:
        with self._cond:
            self._stopping = True
            self._drain_deadline = time.monotonic() + drain_timeout_s
            self._cond.notify_all()
        self._stop_evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=drain_timeout_s + 5.0)

    @property
    def status(self) -> str:
        with self._cond:
            if self._current is not None:
                return "speaking"
            return self._backend_state

    def _on_done(self, uid: int) -> None:
        with self._cond:
            if self._current is not None and self._current.seq == uid:
                self._current_done = True
                self._cond.notify_all()

    def _on_dead(self, reason: str) -> None:
        with self._cond:
            self._backend_dead = reason
            self._cond.notify_all()

    def _default_factory(self) -> TTSBackend:
        if not self._cfg.enable_tts:
            return LogOnlyBackend()
        return PowerShellBackend(rate=self._cfg.tts_rate, startup_timeout_s=self._cfg.tts_startup_timeout_s)

    def _run(self) -> None:
        try:
            self._boot_backend()
            while True:
                action, payload = self._next_action()
                if action == "exit":
                    break
                if action == "restart":
                    self._close_backend()
                    self._boot_backend()
                elif action == "speak":
                    uid, text, prio = payload
                    self._call_backend(lambda: self._backend.speak(uid, text))
                elif action == "cancel":
                    self._call_backend(self._backend.cancel)
        except Exception:
            log.exception("TTS worker crashed")
        finally:
            self._close_backend()
            with self._cond:
                self._backend_state = "stopped"

    def _next_action(self) -> Tuple[str, object]:
        with self._cond:
            while True:
                now = time.monotonic()
                if self._stopping and now >= self._drain_deadline:
                    return "exit", None
                if self._backend_dead is not None:
                    reason, self._backend_dead = self._backend_dead, None
                    self._current = None
                    return "restart", reason
                if self._current is not None:
                    if self._current_done:
                        self._current, self._current_done, self._preempt = None, False, False
                        continue
                    if self._preempt:
                        self._current, self._preempt = None, False
                        return "cancel", None
                    if now >= self._current_deadline:
                        self._current = None
                        return "restart", "utterance timed out"
                    self._cond.wait(min(0.25, self._current_deadline - now))
                    continue
                if self._heap:
                    utt = heapq.heappop(self._heap)
                    if utt.expires_at < now:
                        continue
                    self._current, self._current_done, self._preempt = utt, False, False
                    self._current_deadline = now + 3.0 + 0.12 * len(utt.text)
                    return "speak", (utt.seq, utt.text, utt.priority)
                if self._stopping:
                    return "exit", None
                self._cond.wait(0.5)

    def _call_backend(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            self._on_dead(f"{type(exc).__name__}: {exc}")

    def _boot_backend(self) -> None:
        attempt = 0
        while not self._stop_evt.is_set():
            now = time.monotonic()
            while self._restart_times and now - self._restart_times[0] > 60.0:
                self._restart_times.popleft()
            if len(self._restart_times) >= self._cfg.tts_max_restarts:
                break
            self._restart_times.append(now)
            candidate = self._factory()
            try:
                candidate.start(self._on_done, self._on_dead)
            except TTSUnavailable as exc:
                if exc.permanent:
                    break
            except Exception:
                pass
            else:
                self._set_backend(candidate)
                return
            attempt += 1
            if self._stop_evt.wait(min(8.0, 0.5 * 2 ** attempt)):
                break
        fallback = LogOnlyBackend()
        fallback.start(self._on_done, self._on_dead)
        self._set_backend(fallback)

    def _set_backend(self, backend: TTSBackend) -> None:
        with self._cond:
            self._backend = backend
            self._backend_state = "idle" if backend.name != LogOnlyBackend.name else "log-only"

    def _close_backend(self) -> None:
        try:
            self._backend.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# 4. VOICE COMMAND LISTENER
# ---------------------------------------------------------------------------

class VoiceCommandListener:
    MAX_MIC_REOPENS = 5

    def __init__(self, cfg: AudioConfig, on_command: Callable[[str], None]) -> None:
        self._cfg = cfg
        self._on_command = on_command
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status = "disabled"
        self._rl = RateLimitedLog(log, interval_s=30.0)
        alternatives = "|".join(re.escape(p) for p in cfg.wake_phrases)
        self._pattern = re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)

    @property
    def status(self) -> str:
        return self._status

    def start(self) -> bool:
        try:
            import speech_recognition  # noqa: F401
        except ImportError:
            return False
        self._status = "starting"
        self._thread = threading.Thread(target=self._run, name="mic-listener", daemon=True)
        self._thread.start()
        return True

    def request_stop(self) -> None:
        self._stop.set()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s + self._cfg.mic_recognize_timeout_s)

    def matches(self, utterance: str) -> bool:
        return bool(self._pattern.search(utterance))

    def _run(self) -> None:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = self._cfg.mic_recognize_timeout_s
        recognizer.dynamic_energy_threshold = True

        failures = 0
        while not self._stop.is_set():
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    self._status = "listening"
                    failures = 0
                    self._listen_loop(sr, recognizer, source)
            except (OSError, AttributeError) as exc:
                failures += 1
                self._status = "error"
                if failures >= self.MAX_MIC_REOPENS:
                    break
                self._stop.wait(min(10.0, 2.0 * failures))
            except Exception:
                break
        self._status = "disabled"

    def _listen_loop(self, sr, recognizer, source) -> None:
        while not self._stop.is_set():
            try:
                audio = recognizer.listen(source, timeout=self._cfg.mic_listen_timeout_s,
                                         phrase_time_limit=self._cfg.mic_phrase_limit_s)
            except sr.WaitTimeoutError:
                continue
            if self._stop.is_set():
                return
            try:
                text = recognizer.recognize_google(audio).lower()
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                self._rl.log(logging.WARNING, "request", "Speech recognition unreachable: %s", exc)
                self._stop.wait(2.0)
                continue
            if self.matches(text):
                try:
                    self._on_command("summary")
                except Exception:
                    pass

# ---------------------------------------------------------------------------
# 5. VIDEO SOURCE
# ---------------------------------------------------------------------------

class SourceUnavailable(RuntimeError):
    pass

@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp: float
    index: int

def _is_live(source) -> bool:
    if isinstance(source, int):
        return True
    s = str(source).strip().lower()
    return s.isdigit() or s.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://"))

class VideoSource:
    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        src = cfg.source
        self._source = int(src) if isinstance(src, str) and src.strip().isdigit() else src
        self.is_live = _is_live(self._source)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = 30.0
        self._index = 0
        self._exhausted = False
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._last_returned = -1
        self._stop = threading.Event()
        self._grabber: Optional[threading.Thread] = None

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def open(self) -> None:
        last_err = "unknown error"
        for attempt in range(1, self._cfg.open_retries + 1):
            cap = self._create_capture()
            if cap is not None and cap.isOpened():
                self._cap = cap
                fps = cap.get(cv2.CAP_PROP_FPS)
                self._fps = fps if fps and 1.0 < fps < 240.0 else 30.0
                if self.is_live:
                    self._grabber = threading.Thread(target=self._grab_loop, name="frame-grabber", daemon=True)
                    self._grabber.start()
                return
            if cap is not None:
                cap.release()
            last_err = f"cv2.VideoCapture({self._source!r}) could not be opened"
            if not self.is_live:
                break
            time.sleep(self._cfg.reconnect_backoff_s)
        raise SourceUnavailable(last_err)

    def read(self, timeout_s: float = 2.0) -> Optional[Frame]:
        if self._exhausted or self._cap is None:
            return None
        if not self.is_live:
            return self._read_file()
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while (self._latest is None or self._latest.index == self._last_returned) \
                    and not self._exhausted:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._latest is None or self._latest.index == self._last_returned:
                return None
            self._last_returned = self._latest.index
            return self._latest

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._grabber is not None:
            self._grabber.join(timeout=3.0)
            if self._grabber.is_alive():
                return
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _create_capture(self) -> Optional[cv2.VideoCapture]:
        try:
            if isinstance(self._source, int) and sys.platform == "win32":
                return cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            return cv2.VideoCapture(self._source)
        except cv2.error:
            return None

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        w, h = self._cfg.frame_size
        if image.shape[1] != w or image.shape[0] != h:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return image

    def _read_file(self) -> Optional[Frame]:
        ok, image = self._cap.read()
        if not ok or image is None:
            self._exhausted = True
            return None
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts <= 0.0 and self._index > 0:
            ts = self._index / self._fps
        frame = Frame(self._prepare(image), ts, self._index)
        self._index += 1
        return frame

    def _grab_loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok or image is None:
                failures += 1
                if failures >= self._cfg.max_consecutive_read_failures:
                    if not self._reconnect():
                        break
                    failures = 0
                else:
                    self._stop.wait(0.01)
                continue
            failures = 0
            frame = Frame(self._prepare(image), time.monotonic(), self._index)
            self._index += 1
            with self._cond:
                self._latest = frame
                self._cond.notify_all()
        with self._cond:
            if not self._stop.is_set():
                self._exhausted = True
            self._cond.notify_all()

    def _reconnect(self) -> bool:
        self._cap.release()
        for attempt in range(1, self._cfg.open_retries + 1):
            if self._stop.wait(self._cfg.reconnect_backoff_s * attempt):
                return False
            cap = self._create_capture()
            if cap is not None and cap.isOpened():
                self._cap = cap
                return True
            if cap is not None:
                cap.release()
        return False

# ---------------------------------------------------------------------------
# 6. CAMERA HEALTH MONITOR
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    reason: str
    changed: bool
    brightness: float
    sharpness: float

class CameraHealthMonitor:
    def __init__(self, cfg: HealthConfig) -> None:
        self._cfg = cfg
        self._ok = True
        self._bad_streak = 0
        self._good_streak = 0
        self._reason = "OPERATIONAL"

    @property
    def ok(self) -> bool:
        return self._ok

    @staticmethod
    def measure(image: np.ndarray) -> tuple:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def check(self, image: np.ndarray) -> HealthStatus:
        brightness, sharpness = self.measure(image)
        if brightness < self._cfg.brightness_threshold:
            frame_reason = "Too dark"
        elif sharpness < self._cfg.variance_threshold:
            frame_reason = "Lens blocked"
        else:
            frame_reason = None

        changed = False
        if frame_reason is None:
            self._bad_streak = 0
            self._good_streak += 1
            if not self._ok and self._good_streak >= self._cfg.ok_frames_to_clear:
                self._ok, self._reason, changed = True, "OPERATIONAL", True
        else:
            self._good_streak = 0
            self._bad_streak += 1
            if self._ok and self._bad_streak >= self._cfg.fail_frames_to_trip:
                self._ok, changed = False, True
            if not self._ok:
                self._reason = frame_reason
        return HealthStatus(self._ok, self._reason, changed, brightness, sharpness)

# ---------------------------------------------------------------------------
# 7. DETECTOR (YOLO26s + ByteTrack)
# ---------------------------------------------------------------------------

class DetectorUnavailable(RuntimeError):
    pass

@dataclass(frozen=True)
class Detection:
    track_id: int
    class_id: int
    label: str
    conf: float
    bbox: Tuple[float, float, float, float]

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

class Detector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self._cfg = cfg
        self._model = None
        self.names: dict = {}

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailable("ultralytics is not installed") from exc
        try:
            self._model = YOLO(self._cfg.weights)
            self.names = dict(self._model.names)
            self._model.predict(np.zeros((self._cfg.imgsz, self._cfg.imgsz, 3), np.uint8),
                                imgsz=self._cfg.imgsz, device=self._cfg.device or None, verbose=False)
        except Exception as exc:
            raise DetectorUnavailable(f"failed to load {self._cfg.weights}: {exc}") from exc

    def track(self, image: np.ndarray) -> List[Detection]:
        if self._model is None:
            raise DetectorUnavailable("detector not loaded")
        results = self._model.track(
            source=image,
            persist=True,
            tracker=self._cfg.tracker,
            classes=list(self._cfg.classes),
            conf=self._cfg.conf,
            imgsz=self._cfg.imgsz,
            device=self._cfg.device or None,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()
        cls = boxes.cls.int().cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [
            Detection(int(t), int(c), self.names.get(int(c), str(int(c))), float(p),
                      tuple(float(v) for v in b))
            for b, t, c, p in zip(xyxy, ids, cls, conf)
        ]

    def reset_tracker(self) -> None:
        predictor = getattr(self._model, "predictor", None)
        for trk in getattr(predictor, "trackers", None) or []:
            try:
                trk.reset()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# 8. PROXIMITY ESTIMATOR & KINEMATICS
# ---------------------------------------------------------------------------

_EDGE_MARGIN_PX = 2.0

class ThreatLevel(IntEnum):
    UNKNOWN = auto()
    RECEDING = auto()
    STABLE = auto()
    APPROACHING = auto()
    CRITICAL = auto()

@dataclass(frozen=True)
class ThreatAssessment:
    track_id: int
    level: ThreatLevel
    distance_m: Optional[float]
    ttc_s: Optional[float]
    lateral_m: Optional[float]
    lateral_vel_mps: float      # <-- Added for Prediction Vector HUD
    on_course: bool
    is_critical: bool
    newly_critical: bool
    description: str

@dataclass
class _Sample:
    t: float
    log_h: Optional[float]
    log_w: Optional[float]
    distance: Optional[float]
    lateral: Optional[float]

@dataclass
class _Track:
    first_seen: float
    last_seen: float
    samples: Deque[_Sample] = field(default_factory=lambda: deque(maxlen=90))
    confs: Deque[float] = field(default_factory=lambda: deque(maxlen=15))
    distance_ema: Optional[float] = None
    anchor: Optional[Tuple[float, float, float]] = None
    hits: Deque[bool] = field(default_factory=deque)
    misses: int = 0
    armed: bool = False

@dataclass(frozen=True)
class _Fit:
    slope: float
    tstat: float
    value_at_end: float

def _linear_fit(ts: np.ndarray, ys: np.ndarray) -> Optional[_Fit]:
    n = len(ts)
    if n < 3:
        return None
    t_mean, y_mean = ts.mean(), ys.mean()
    dt = ts - t_mean
    sxx = float(np.dot(dt, dt))
    if sxx <= 1e-9:
        return None
    slope = float(np.dot(dt, ys - y_mean)) / sxx
    resid = ys - (y_mean + slope * dt)
    se = math.sqrt(float(np.dot(resid, resid)) / (n - 2) / sxx)
    tstat = slope / se if se > 1e-12 else math.copysign(1e9, slope)
    return _Fit(slope, tstat, y_mean + slope * (ts[-1] - t_mean))

class ProximityEstimator:
    def __init__(self, cfg: ProximityConfig, cam: CameraConfig) -> None:
        self._cfg = cfg
        self._w, self._h = cam.frame_size
        self.focal_px = (self._w / 2.0) / math.tan(math.radians(cam.horizontal_fov_deg) / 2.0)
        self._cx0 = self._w / 2.0
        self._horizon_y = cam.horizon_ratio * self._h
        self._cam_h = cam.mount_height_m
        span = self._h - self._horizon_y
        self._bottom_edge_z = self.focal_px * self._cam_h / span if span > 0 else None
        self._tracks: Dict[int, _Track] = {}

    def estimate_distance(self, det: Detection) -> Optional[float]:
        estimate, bound = self._measure(det)
        if estimate is None:
            return bound
        return estimate if bound is None else min(estimate, bound)

    def _measure(self, det: Detection) -> Tuple[Optional[float], Optional[float]]:
        x1, y1, x2, y2 = det.bbox
        h_px = y2 - y1
        clip_top = y1 <= _EDGE_MARGIN_PX
        clip_bottom = y2 >= self._h - _EDGE_MARGIN_PX
        f = self.focal_px

        estimates: List[Tuple[float, float]] = []
        bounds: List[float] = []

        real_h = self._cfg.class_heights_m.get(det.class_id)
        if real_h and h_px > 1.0:
            z = f * real_h / h_px
            if clip_top or clip_bottom:
                bounds.append(z)
            else:
                estimates.append((z, 0.20 * z + 0.10))

        if clip_bottom:
            if self._bottom_edge_z:
                bounds.append(self._bottom_edge_z)
        elif y2 > self._horizon_y + 4.0:
            z = f * self._cam_h / (y2 - self._horizon_y)
            sigma = z * z * self._cfg.ground_pitch_err_px / (f * self._cam_h) + 0.05 * z
            estimates.append((z, sigma))

        fused = None
        if estimates:
            weights = [1.0 / (s * s) for _, s in estimates]
            fused = sum(z * w for (z, _), w in zip(estimates, weights)) / sum(weights)
        return fused, (min(bounds) if bounds else None)

    def update(self, det: Detection, t: float) -> ThreatAssessment:
        cfg = self._cfg
        trk = self._tracks.get(det.track_id)
        if trk is None or t <= trk.last_seen:
            trk = _Track(first_seen=t, last_seen=t, hits=deque(maxlen=cfg.enter_window))
            self._tracks[det.track_id] = trk
        trk.last_seen = t
        trk.confs.append(det.conf)

        x1, y1, x2, y2 = det.bbox
        clip_v = y1 <= _EDGE_MARGIN_PX or y2 >= self._h - _EDGE_MARGIN_PX
        clip_hz = x1 <= _EDGE_MARGIN_PX or x2 >= self._w - _EDGE_MARGIN_PX
        h_px, w_px = max(y2 - y1, 1.0), max(x2 - x1, 1.0)
        log_h = None if clip_v else math.log(h_px)
        log_w = None if clip_hz else math.log(w_px)

        distance, bound = self._measure(det)
        if distance is not None and log_h is not None and log_w is not None:
            trk.anchor = (distance, log_h, log_w)
        elif distance is None and trk.anchor is not None:
            z_a, lh_a, lw_a = trk.anchor
            if log_h is not None:
                distance = z_a * math.exp(lh_a - log_h)
            elif log_w is not None:
                distance = z_a * math.exp(lw_a - log_w)
        if bound is not None:
            distance = bound if distance is None else min(distance, bound)
        if distance is not None:
            a = cfg.ema_alpha
            trk.distance_ema = distance if trk.distance_ema is None else \
                a * distance + (1 - a) * trk.distance_ema
        lateral = None
        if distance is not None:
            lateral = distance * (det.center_x - self._cx0) / self.focal_px

        trk.samples.append(_Sample(t=t, log_h=log_h, log_w=log_w, distance=distance, lateral=lateral))
        return self._assess(det, trk, t)

    def prune(self, t: float) -> None:
        cutoff = t - self._cfg.track_timeout_s
        for tid in [k for k, v in self._tracks.items() if v.last_seen < cutoff]:
            del self._tracks[tid]

    def reset(self) -> None:
        self._tracks.clear()

    def _window(self, trk: _Track, t: float) -> List[_Sample]:
        start = t - self._cfg.window_s
        return [s for s in trk.samples if s.t >= start]

    def _expansion(self, window: List[_Sample]) -> Optional[_Fit]:
        for attr in ("log_h", "log_w"):
            pts = [(s.t, getattr(s, attr)) for s in window if getattr(s, attr) is not None]
            if len(pts) >= self._cfg.min_samples:
                arr = np.asarray(pts, dtype=np.float64)
                return _linear_fit(arr[:, 0], arr[:, 1])
        return None

    def _assess(self, det: Detection, trk: _Track, t: float) -> ThreatAssessment:
        cfg = self._cfg
        window = self._window(trk, t)
        age = t - trk.first_seen
        distance = trk.distance_ema

        fit = self._expansion(window)
        ttc: Optional[float] = None
        closing = receding = False
        if fit is not None:
            if fit.slope > 1e-6 and fit.tstat >= cfg.min_expansion_tstat:
                ttc, closing = 1.0 / fit.slope, True
            elif fit.tstat <= -cfg.min_expansion_tstat:
                receding = True

        lat_pts = [(s.t, s.lateral) for s in window if s.lateral is not None]
        lateral_now: Optional[float] = None
        lateral_vel = 0.0
        if lat_pts:
            arr = np.asarray(lat_pts, dtype=np.float64)
            lat_fit = _linear_fit(arr[:, 0], arr[:, 1]) if len(lat_pts) >= cfg.min_samples else None
            if lat_fit is not None:
                lateral_now, lateral_vel = lat_fit.value_at_end, lat_fit.slope
            else:
                lateral_now = float(arr[-1, 1])

        horizon = min(ttc, 3.0) if ttc is not None else 0.0
        on_course = lateral_now is not None and \
            abs(lateral_now + lateral_vel * horizon) <= cfg.corridor_half_width_m
        in_corridor = lateral_now is not None and abs(lateral_now) <= cfg.corridor_half_width_m

        mature = age >= cfg.min_track_age_s and len(window) >= cfg.min_samples
        confident = (sum(trk.confs) / len(trk.confs)) >= cfg.emergency_min_conf
        big_enough = (det.bbox[3] - det.bbox[1]) >= cfg.min_box_height_px
        immediate = distance is not None and distance <= cfg.immediate_distance_m and in_corridor
        imminent = (closing and big_enough and ttc is not None and ttc <= cfg.critical_ttc_s
                    and distance is not None and distance <= cfg.critical_distance_m and on_course)
        candidate = mature and confident and (immediate or imminent)

        trk.hits.append(candidate)
        newly = False
        if trk.armed:
            trk.misses = 0 if candidate else trk.misses + 1
            if trk.misses >= cfg.exit_misses:
                trk.armed, trk.misses = False, 0
                trk.hits.clear()
        elif sum(trk.hits) >= cfg.enter_hits:
            trk.armed, trk.misses, newly = True, 0, True

        if not mature:
            level, text = ThreatLevel.UNKNOWN, "TRACKING..."
        elif trk.armed:
            level = ThreatLevel.CRITICAL
            text = "CRITICAL: CLOSING IN" if closing else "CRITICAL: OBSTACLE"
        elif closing and ttc is not None and ttc <= cfg.approach_ttc_s:
            level = ThreatLevel.APPROACHING
            if on_course:
                text = "APPROACHING"
            else:
                text = "APPROACHING FROM LEFT" if lateral_vel > 0 else "APPROACHING FROM RIGHT"
        elif receding:
            level, text = ThreatLevel.RECEDING, "RECEDING"
        else:
            level, text = ThreatLevel.STABLE, "STABLE"

        return ThreatAssessment(
            track_id=det.track_id, level=level, distance_m=distance, ttc_s=ttc,
            lateral_m=lateral_now, lateral_vel_mps=lateral_vel, on_course=on_course, 
            is_critical=trk.armed, newly_critical=newly, description=text,
        )

# ---------------------------------------------------------------------------
# 9. ALERT POLICY & SUMMARIES
# ---------------------------------------------------------------------------

SECTORS = ("left", "ahead", "right")
_SECTOR_PHRASE = {"left": "left", "ahead": "ahead", "right": "right"}
_SECTOR_TITLE = {"left": "On your left", "ahead": "Ahead", "right": "On your right"}
_IRREGULAR_PLURALS = {"person": "people", "bus": "buses"}

def sector_of(center_x: float, frame_w: int) -> str:
    if center_x < frame_w / 3.0:
        return "left"
    if center_x > 2.0 * frame_w / 3.0:
        return "right"
    return "ahead"

def _count_phrase(label: str, n: int) -> str:
    if n == 1:
        return f"{'an' if label[:1] in 'aeiou' else 'a'} {label}"
    return f"{n} {_IRREGULAR_PLURALS.get(label, label + 's')}"

def build_summary(scene: Dict[str, Counter]) -> str:
    parts = []
    for sector in SECTORS:
        counts = scene.get(sector)
        desc = ", ".join(_count_phrase(l, n) for l, n in counts.most_common()) if counts else "clear"
        parts.append(f"{_SECTOR_TITLE[sector]}: {desc}.")
    return "Status update. " + " ".join(parts)

def _distance_phrase(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return ""
    if distance_m < 1.5:
        return ", close!"
    return f", {round(distance_m)} meters"

class AlertPolicy:
    def __init__(self, cfg: AlertConfig) -> None:
        self._cfg = cfg
        self._track_last: Dict[int, float] = {}
        self._zone_last: Dict[Tuple[str, str], Tuple[float, int]] = {}

    def consider(self, det: Detection, threat: ThreatAssessment, sector: str, now: float) -> Optional[str]:
        if not threat.is_critical:
            return None
        last = self._track_last.get(det.track_id)
        if last is not None and now - last < self._cfg.track_cooldown_s:
            return None
        zone = (det.label, sector)
        zone_hit = self._zone_last.get(zone)
        if zone_hit is not None and zone_hit[1] != det.track_id \
                and now - zone_hit[0] < self._cfg.zone_cooldown_s:
            self._track_last[det.track_id] = zone_hit[0]
            return None
        self._track_last[det.track_id] = now
        self._zone_last[zone] = (now, det.track_id)
        return f"{det.label} {_SECTOR_PHRASE[sector]}{_distance_phrase(threat.distance_m)}"

    def prune(self, now: float) -> None:
        horizon = 2.0 * max(self._cfg.track_cooldown_s, self._cfg.zone_cooldown_s)
        self._track_last = {k: v for k, v in self._track_last.items() if now - v < horizon}
        self._zone_last = {k: v for k, v in self._zone_last.items() if now - v[0] < horizon}

# ---------------------------------------------------------------------------
# 10. HUD DRAWING UTILITIES & RADAR
# ---------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX
LEVEL_COLORS = {
    ThreatLevel.UNKNOWN: (200, 200, 200),
    ThreatLevel.RECEDING: (100, 255, 100),
    ThreatLevel.STABLE: (0, 255, 0),
    ThreatLevel.APPROACHING: (0, 140, 255),
    ThreatLevel.CRITICAL: (0, 0, 255),
}

def draw_radar(img: np.ndarray, threats: List[ThreatAssessment]) -> None:
    """Draws a 2D 'Spider-Sense' spatial map of all objects."""
    RADAR_RADIUS = 90
    RADAR_CENTER = (img.shape[1] - RADAR_RADIUS - 30, img.shape[0] - RADAR_RADIUS - 30)
    MAX_DIST = 15.0  # Range of the radar in meters

    # Dark background glass overlay
    overlay = img.copy()
    cv2.circle(overlay, RADAR_CENTER, RADAR_RADIUS, (15, 25, 15), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Concentric distance rings and crosshairs
    cv2.circle(img, RADAR_CENTER, RADAR_RADIUS, (0, 200, 0), 2)
    cv2.circle(img, RADAR_CENTER, RADAR_RADIUS // 2, (0, 100, 0), 1)
    cv2.circle(img, RADAR_CENTER, RADAR_RADIUS // 3, (0, 100, 0), 1)
    cv2.line(img, (RADAR_CENTER[0], RADAR_CENTER[1] - RADAR_RADIUS), (RADAR_CENTER[0], RADAR_CENTER[1] + RADAR_RADIUS), (0, 100, 0), 1)
    cv2.line(img, (RADAR_CENTER[0] - RADAR_RADIUS, RADAR_CENTER[1]), (RADAR_CENTER[0] + RADAR_RADIUS, RADAR_CENTER[1]), (0, 100, 0), 1)

    # Draw the user in the center
    cv2.circle(img, RADAR_CENTER, 4, (255, 255, 255), -1)

    for t in threats:
        if t.distance_m is None or t.lateral_m is None:
            continue
        
        # Map 3D real-world coordinates to 2D radar pixels
        z = min(t.distance_m, MAX_DIST)
        x = t.lateral_m 
        
        scale = RADAR_RADIUS / MAX_DIST
        px = int(RADAR_CENTER[0] + (x * scale))
        py = int(RADAR_CENTER[1] - (z * scale))  # Subtract Z because forward is 'up' on the screen
        
        # Draw dot if inside radar bounds
        if math.hypot(px - RADAR_CENTER[0], py - RADAR_CENTER[1]) <= RADAR_RADIUS:
            color = LEVEL_COLORS.get(t.level, (200, 200, 200))
            if t.is_critical and int(time.time() * 6) % 2 == 0:
                cv2.circle(img, (px, py), 8, (0, 0, 255), -1)
                cv2.circle(img, (px, py), 12, (0, 0, 255), 1) # Flashing pulse effect
            else:
                cv2.circle(img, (px, py), 5, color, -1)

def draw_detection(img: np.ndarray, det: Detection, threat: ThreatAssessment) -> None:
    color = LEVEL_COLORS[threat.level]
    x1, y1, x2, y2 = (int(v) for v in det.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if threat.is_critical else 2)
    cv2.putText(img, f"ID:{det.track_id} {det.label} ({det.conf:.2f})",
                (x1, max(30, y1 - 28)), FONT, 0.55, color, 2)
    
    # ---------------------------------------------------------
    # NEW: Kinematic Danger Zone Vector Arrows
    # ---------------------------------------------------------
    if abs(threat.lateral_vel_mps) > 0.5:
        bx_center = (x1 + x2) // 2
        by_bottom = y2
        arrow_len = int(threat.lateral_vel_mps * 40)
        end_pt = (bx_center + arrow_len, by_bottom - 10)
        cv2.arrowedLine(img, (bx_center, by_bottom - 10), end_pt, (0, 255, 255), 3, tipLength=0.3)

    detail = threat.description
    if threat.distance_m is not None:
        detail += f" {threat.distance_m:.1f}m"
    if threat.ttc_s is not None:
        detail += f" TTC {threat.ttc_s:.1f}s"
    cv2.putText(img, detail, (x1, max(48, y1 - 8)), FONT, 0.5, color, 2)

def draw_status(img: np.ndarray, fps: float, tts_status: str, mic_status: str) -> None:
    cv2.putText(img, "SYSTEM: SILENT GUARD ACTIVE (A = status, Q = quit)", (20, 35),
                FONT, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"{fps:4.1f} FPS | TTS: {tts_status} | MIC: {mic_status}",
                (20, img.shape[0] - 15), FONT, 0.5, (255, 255, 255), 1)

def draw_fault(img: np.ndarray, reason: str) -> None:
    cv2.putText(img, f"SENSOR FAULT: {reason} - alerts paused", (30, 80),
                FONT, 0.9, (0, 0, 255), 2)

# ---------------------------------------------------------------------------
# 11. MAIN APPLICATION ORCHESTRATION
# ---------------------------------------------------------------------------

EXIT_OK, EXIT_NO_SOURCE, EXIT_NO_MODEL, EXIT_RUNTIME = 0, 2, 3, 4
MAX_CONSECUTIVE_INFERENCE_ERRORS = 25

class SilentGuardApp:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._stop = threading.Event()
        self._summary_requested = threading.Event()
        self._headless = cfg.headless
        self.tts = TTSService(cfg.audio)
        self.listener = VoiceCommandListener(cfg.audio, on_command=lambda _cmd: self._summary_requested.set())
        self.detector = Detector(cfg.detector)
        self.source = VideoSource(cfg.camera)
        self.health = CameraHealthMonitor(cfg.health)
        self.proximity = ProximityEstimator(cfg.proximity, cfg.camera)
        self.policy = AlertPolicy(cfg.alerts)
        self._rl = RateLimitedLog(log, interval_s=5.0)
        self._fps = 0.0
        self._inference_errors = 0

    def run(self) -> int:
        self._install_signal_handlers()
        self.tts.start()
        if self.cfg.audio.enable_mic:
            self.listener.start()
        try:
            try:
                self.detector.load()
            except DetectorUnavailable as exc:
                log.critical("Vision model unavailable: %s", exc)
                self.tts.say("Vision system failed to start.", Priority.EMERGENCY, ttl_s=10.0)
                return EXIT_NO_MODEL
            try:
                self.source.open()
            except SourceUnavailable as exc:
                log.critical("Video source unavailable: %s", exc)
                self.tts.say("Camera not available.", Priority.EMERGENCY, ttl_s=10.0)
                return EXIT_NO_SOURCE
            self._init_window()
            self.tts.say("Silent guard active.", Priority.INFO)
            return self._loop()
        except KeyboardInterrupt:
            return EXIT_OK
        except Exception:
            log.exception("Fatal error in main loop")
            self.tts.say("Guard system error. Alerts stopped.", Priority.EMERGENCY, ttl_s=10.0)
            return EXIT_RUNTIME
        finally:
            self._shutdown()

    def _loop(self) -> int:
        last = time.monotonic()
        frame_idx = 1
        FRAME_SKIP = 1  # Retained at 1 to fully utilize the RTX 3050

        while not self._stop.is_set():
            frame = self.source.read(timeout_s=2.0)
            if frame is None:
                if self.source.exhausted:
                    if self.source.is_live:
                        log.error("Camera lost; stopping")
                        self.tts.say("Camera lost. Alerts stopped.", Priority.EMERGENCY, ttl_s=10.0)
                        return EXIT_NO_SOURCE
                    return EXIT_OK
                self._rl.log(logging.WARNING, "no-frame", "No frame for 2 s; waiting for camera")
                if not self._pump_ui(None):
                    break
                continue

            frame_idx += 1
            if frame_idx % FRAME_SKIP == 0:
                self._process(frame)

            now = time.monotonic()
            dt, last = now - last, now
            if dt > 0:
                self._fps = 1.0 / dt if self._fps == 0 else 0.9 * self._fps + 0.1 / dt
            if not self._pump_ui(frame.image):
                break
        return EXIT_OK

    def _process(self, frame: Frame) -> None:
        img = frame.image
        health = self.health.check(img)
        if health.changed:
            self._on_health_change(health.ok, health.reason)
        if not health.ok:
            draw_fault(img, health.reason)
            if self._summary_requested.is_set():
                self._summary_requested.clear()
                self.tts.say("Camera view blocked. I cannot see around you.", Priority.SUMMARY,
                            ttl_s=self.cfg.alerts.summary_ttl_s)
            return

        try:
            detections = self.detector.track(img)
            self._inference_errors = 0
        except Exception as exc:
            self._inference_errors += 1
            self._rl.log(logging.ERROR, "inference", "Inference failed (%d consecutive): %s",
                         self._inference_errors, exc)
            if self._inference_errors >= MAX_CONSECUTIVE_INFERENCE_ERRORS:
                raise RuntimeError("inference is failing persistently") from exc
            return

        now = time.monotonic()
        scene: Dict[str, Counter] = {s: Counter() for s in SECTORS}
        frame_w = img.shape[1]
        active_threats = []  # Collector for the radar HUD
        
        for det in detections:
            threat = self.proximity.update(det, frame.timestamp)
            active_threats.append(threat)
            sector = sector_of(det.center_x, frame_w)
            scene[sector][det.label] += 1
            phrase = self.policy.consider(det, threat, sector, now)
            if phrase:
                log.warning("[PROXIMITY ALARM] %s", phrase)
                self.tts.say(phrase, Priority.EMERGENCY, ttl_s=self.cfg.alerts.emergency_ttl_s)
            draw_detection(img, det, threat)

        # Draw the new Spider-Sense Radar after processing all threats
        draw_radar(img, active_threats)

        self.proximity.prune(frame.timestamp)
        self.policy.prune(now)

        if self._summary_requested.is_set():
            self._summary_requested.clear()
            summary = build_summary(scene)
            log.info("[SUMMARY] %s", summary)
            self.tts.say(summary, Priority.SUMMARY, ttl_s=self.cfg.alerts.summary_ttl_s)

    def _on_health_change(self, ok: bool, reason: str) -> None:
        if ok:
            self.tts.say("Camera view restored.", Priority.SUMMARY, ttl_s=5.0)
        else:
            self.tts.say(f"Warning: camera {reason.lower()}. Alerts paused.", Priority.SUMMARY, ttl_s=5.0)
            self.proximity.reset()
            self.detector.reset_tracker()

    def _init_window(self) -> None:
        if self._headless:
            return
        try:
            cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            self._headless = True

    def _pump_ui(self, image: Optional[np.ndarray]) -> bool:
        if self._headless:
            return True
        if image is not None:
            draw_status(image, self._fps, self.tts.status, self.listener.status)
            cv2.imshow(self.cfg.window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key == ord("a"):
            self._summary_requested.set()
        try:
            if cv2.getWindowProperty(self.cfg.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except cv2.error:
            return False
        return True

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def handler(signum, _frame):
            if self._stop.is_set():
                raise KeyboardInterrupt
            self._stop.set()

        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass

    def _shutdown(self) -> None:
        steps = (
            ("microphone", self.listener.request_stop),
            ("video source", self.source.close),
            ("display", self._close_window),
            ("speech", lambda: self.tts.stop(drain_timeout_s=5.0)),
            ("microphone join", self.listener.stop),
        )
        for _, step in steps:
            try:
                step()
            except Exception:
                pass

    def _close_window(self) -> None:
        if not self._headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

def _fmt(value: Optional[float], unit: str) -> str:
    return "n/a" if value is None else f"{value:.1f}{unit}"

# ---------------------------------------------------------------------------
# 12. ENTRY POINT
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> AppConfig:
    d = AppConfig()
    p = argparse.ArgumentParser(prog="silent_guard", description="Silent-by-default hazard assistant")
    p.add_argument("--source", default=str(d.camera.source), help="webcam index, file path or stream URL")
    p.add_argument("--weights", default=d.detector.weights)
    p.add_argument("--conf", type=float, default=d.detector.conf)
    p.add_argument("--imgsz", type=int, default=d.detector.imgsz)
    p.add_argument("--device", default=d.detector.device)
    p.add_argument("--hfov", type=float, default=d.camera.horizontal_fov_deg)
    p.add_argument("--cam-height", type=float, default=d.camera.mount_height_m)
    p.add_argument("--horizon", type=float, default=d.camera.horizon_ratio)
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--no-mic", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--log-level", default=d.log_level, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", default=d.log_file)
    a = p.parse_args(argv)

    source = int(a.source) if a.source.strip().isdigit() else a.source
    return AppConfig(
        camera=CameraConfig(source=source, horizontal_fov_deg=a.hfov, mount_height_m=a.cam_height, horizon_ratio=a.horizon),
        detector=DetectorConfig(weights=a.weights, conf=a.conf, imgsz=a.imgsz, device=a.device),
        audio=AudioConfig(enable_tts=not a.no_tts, enable_mic=not a.no_mic),
        headless=a.headless,
        log_level=a.log_level,
        log_file=a.log_file,
    )

def main(argv=None) -> int:
    cfg = parse_args(argv)
    setup_logging(cfg.log_level, cfg.log_file or None)
    log.info("Starting Silent Guard: source=%r weights=%s", cfg.camera.source, cfg.detector.weights)
    return SilentGuardApp(cfg).run()

if __name__ == "__main__":
    sys.exit(main())
