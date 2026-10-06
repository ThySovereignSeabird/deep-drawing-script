"""
iDraw / GRBL plotter controller
================================
"""

import sys
import re
import math
import json
import time
import queue
import threading
import random
from dataclasses import dataclass
from typing import Optional, List

import serial
import requests


# ------------------------- Settings -------------------------------------

PORT = "/dev/cu.usbmodem201912341"
BAUD = 115200
DATASET_URL = "https://storage.googleapis.com/quickdraw_dataset/full/raw/"
SPEED_MULTIPLIER = 20  # 1.0 = real time, higher value = slower drawing

X_MAX = 600
Y_MAX = 600
BOUNDS = 590
PEN_UP_Z = 0
PEN_DOWN_Z = -8

DISABLE_PEN = False
if DISABLE_PEN:
    PEN_DOWN_Z = PEN_UP_Z

GRBL_RX_BUFFER_SIZE = 128  # bytes; GRBL default serial RX buffer. Match build's config.h.
STATUS_POLL_HZ = 20.0  # position sample rate. independent of the G-code send rate.
ONLY_SAMPLE_PEN_DOWN = True  # gate logging to when pen is down
PEN_DOWN_Z_THRESHOLD = -0.5  # any commanded Z <= this counts as "down"

HAS_LIMIT_SWITCHES = True  # flip to True if we implement iDraw endstops
CRASH_HOME_FEEDRATE = 1500  # mm/min -- keep slow for a repeatable stop
CRASH_HOME_OVERTRAVEL = 50  # mm past nominal travel, guarantees contact

# Data augmentation gates
SCALE_DATA = True
SHEAR_DATA = True

WORDS_TO_PLOT = ["dragon"]  # add more Quick, Draw! filenames to queue
                            # homes between drawings

# ------------------------------------------------------------------------


@dataclass
class PositionSample:
    """Position sample."""
    t: float
    x: float
    y: float
    z: float
    pen_down: bool


class PositionLog:
    """Thread-safe log containing position samples. Append-only."""

    def __init__(self):
        self._lock = threading.Lock()
        self._samples: List[PositionSample] = []

    def add(self, sample: PositionSample):
        with self._lock:
            self._samples.append(sample)

    def snapshot(self) -> List[PositionSample]:
        with self._lock:
            return list(self._samples)

    def save_csv(self, path: str):
        with open(path, "w") as f:
            f.write("t,x,y,z,pen_down\n")
            for s in self.snapshot():
                f.write(f"{s.t:.4f},{s.x:.3f},{s.y:.3f},{s.z:.3f},{int(s.pen_down)}\n")


class Stopwatch:
    def __init__(self):
        self._start_time = time.monotonic()

    def reset(self):
        """Resets the timer back to zero."""
        self._start_time = time.monotonic()

    def elapsed(self):
        """Returns the seconds elapsed since init or last reset."""
        return time.monotonic() - self._start_time


class Augmenter:
    """Scales coordinate values given mins, maxes, and margins."""

    def __init__(self, min_x, max_x, min_y, max_y, bounds=590):
        x_range = max_x - min_x
        y_range = max_y - min_y

        master_scale = bounds / max(x_range, y_range)
        self.smaller_scale = random.uniform(0.25, 0.75)

        final_w = x_range * master_scale * self.smaller_scale
        final_h = y_range * master_scale * self.smaller_scale

        max_offset_x = bounds - final_w
        max_offset_y = bounds - final_h

        self.offset_x = random.uniform(0, max_offset_x)
        self.offset_y = random.uniform(0, max_offset_y)

        self.min_x = min_x
        self.min_y = min_y
        self.multiplier = master_scale * self.smaller_scale

    def scale(self, x, y):
        new_x = (x - self.min_x) * self.multiplier + self.offset_x + 5
        new_y = -((y - self.min_y) * self.multiplier + self.offset_y + 5)
        return new_x, new_y


class MockSerialPort:
    """
    Stand-in for testing without a plotter attached. Immediately echoes
    'ok' to every write so the streaming/ack logic still exercises
    correctly.
    
    No real status reports, so position logging will be empty in mock mode
    """

    def write(self, data):
        text = data.decode(errors="ignore")
        if text != "?":
            print(f"[SERIAL SEND] {text.strip()}")
        return len(data)

    def readline(self):
        time.sleep(0.01)
        return b"ok\n"

    @property
    def in_waiting(self):
        return 0

    def close(self):
        print("[SERIAL] Connection closed.")


# ------------------------- GRBL controller ---------------------------------

STATUS_RE = re.compile(r"(?:MPos|WPos):(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*)")


class GRBLController:
    def __init__(self, port, baud, position_log: Optional[PositionLog] = None,
                 status_hz: float = 0.0, only_log_pen_down: bool = True,
                 pen_down_z_threshold: float = PEN_DOWN_Z_THRESHOLD,
                 mock: bool = False):
        self.mock = mock
        self.position_log = position_log
        self.status_hz = status_hz
        self.only_log_pen_down = only_log_pen_down
        self.pen_down_z_threshold = pen_down_z_threshold

        self._pen_down = False
        self._pen_lock = threading.Lock()

        self._machine_state = "Unknown"
        self._state_lock = threading.Lock()

        self._write_lock = threading.Lock()
        self._ok_queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()

        if mock:
            self.ser = MockSerialPort()
        else:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._poller_thread = None
        if self.status_hz > 0 and self.position_log is not None:
            self._poller_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poller_thread.start()

    def _set_pen_state(self, z_target: float):
        with self._pen_lock:
            self._pen_down = z_target <= self.pen_down_z_threshold

    def _is_pen_down(self) -> bool:
        with self._pen_lock:
            return self._pen_down

    def _track_pen_from_line(self, line: str):
        upper = line.upper()
        if upper.startswith(("G0", "G1")):
            m = re.search(r"Z(-?\d+\.?\d*)", upper)
            if m:
                self._set_pen_state(float(m.group(1)))

    # reader thread: sole owner of ser.readline()
    def _reader_loop(self):
        while not self._stop_event.is_set():
            try:
                raw = self.ser.readline()
            except Exception:
                continue
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
            if not line:
                continue

            if line.startswith("<"):
                self._handle_status_report(line)
            elif "ok" in line.lower() or "error" in line.lower() or "alarm" in line.lower():
                self._ok_queue.put(line)
            else:
                print(f"[{line}]")  # startup banner, $$ dumps, etc.

    def _handle_status_report(self, line: str):
        # e.g. <Idle|MPos:0.000,0.000,0.000|FS:0,0>
        state = line[1:].split("|")[0]
        with self._state_lock:
            self._machine_state = state

        m = STATUS_RE.search(line)
        if not m or self.position_log is None:
            return

        x, y, z = (float(v) for v in m.groups())
        pen_down = self._is_pen_down()
        if self.only_log_pen_down and not pen_down:
            return
        self.position_log.add(PositionSample(time.monotonic(), x, y, z, pen_down))

    # ---------------- poller thread: real-time '?' at a fixed rate --------

    def _poll_loop(self):
        period = 1.0 / self.status_hz
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            with self._write_lock:
                self.ser.write(b"?")
            next_t += period
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.monotonic()  # we're behind; resync, don't drift

    # ---------------- sending: batched streaming + single-line -----------

    def stream(self, lines: List[str]):
        """
        Send a batch of G-code lines using GRBL's character-counting
        streaming protocol. Keep the RX buffer as full as possible instead
        of waiting for 'ok' after every line, so GRBL can plan motion
        across command boundaries.
        """
        pending_lens: List[int] = []
        i = 0
        while i < len(lines) or pending_lens:
            if i < len(lines):
                line = lines[i].strip()
                encoded_len = len(line) + 1  # + newline
                if sum(pending_lens) + encoded_len <= GRBL_RX_BUFFER_SIZE:
                    self._track_pen_from_line(line)
                    with self._write_lock:
                        self.ser.write((line + "\n").encode())
                    print(">>", line)
                    pending_lens.append(encoded_len)
                    i += 1
                    continue
            resp = self._ok_queue.get()
            if "error" in resp.lower() or "alarm" in resp.lower():
                print(f"!!! MACHINE ERROR: {resp}")
            if pending_lens:
                pending_lens.pop(0)

    def send_line(self, cmd: str):
        """Send a single line and block for its ack. Use for setup/config."""
        self._track_pen_from_line(cmd)
        with self._write_lock:
            self.ser.write((cmd + "\n").encode())
        print(">>", cmd)
        while True:
            resp = self._ok_queue.get()
            print(f"[{resp}]")
            if "ok" in resp.lower():
                break
            if "error" in resp.lower() or "alarm" in resp.lower():
                print(f"!!! MACHINE ERROR: {resp}")
                break

    def soft_reset_and_unlock(self):
        print("Sending soft reset and unlock...")
        with self._write_lock:
            self.ser.write(b"\x18")  # Ctrl+X
        time.sleep(2)
        self.send_line("$X")
        print("Plotter unlocked.")

    # ---------------- homing ------------------------------------------

    def wait_idle(self, timeout: float = 60.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            with self._write_lock:
                self.ser.write(b"?")
            with self._state_lock:
                state = self._machine_state
            if state == "Idle":
                return
            time.sleep(0.1)
        print("WARNING: wait_idle() timed out; machine state:", self._machine_state)

    def home(self):
        if HAS_LIMIT_SWITCHES:
            print("Homing ($H)...")
            self.send_line("$H")
            self.wait_idle()
        else:
            self._crash_home()

    def _crash_home(self):
        """
        No limit switches configured: drive slowly into the physical corner
        stops until the motors stall against the frame, then re-zero there.
        This is a workaround, not a substitute for real endstop homing --
        it's still open-loop, so a skipped step during the crash move can
        still leave the new "zero" slightly off. Kept slow on purpose to
        make the stop as repeatable as possible.
        """
        print("Crash-homing (no limit switches configured)...")
        self.send_line(f"G1 Z{PEN_UP_Z} F7000")
        self.send_line("G91")  # relative positioning
        self.send_line(
            f"G1 X{-(CRASH_HOME_OVERTRAVEL + X_MAX)} "
            f"Y{-(CRASH_HOME_OVERTRAVEL + Y_MAX)} F{CRASH_HOME_FEEDRATE}"
        )
        self.wait_idle()
        self.send_line("G90")  # back to absolute
        self.send_line("G10 L20 P1 X0 Y0 Z0")
        print("Crash-home complete, origin re-zeroed.")

    def close(self):
        self._stop_event.set()
        time.sleep(0.1)
        self.ser.close()


# ------------------------- G-code generation --------------------------------

def build_stroke_gcode(stroke, augmenter: Augmenter) -> List[str]:
    lines = []
    x_coords, y_coords, timestamps = stroke

    lines.append(f"G1 Z{PEN_UP_Z} F7000")
    start_x, start_y = augmenter.scale(x_coords[0], y_coords[0])
    lines.append(f"G0 X{start_x:.3f} Y{start_y:.3f} F11500")
    lines.append(f"G1 Z{PEN_DOWN_Z} F7000")

    for i in range(1, len(x_coords)):
        delta_ms = timestamps[i] - timestamps[i - 1]
        delay_s = delta_ms / 1000.0
        if delay_s <= 0:
            continue

        x, y = augmenter.scale(x_coords[i], y_coords[i])
        x_prior, y_prior = augmenter.scale(x_coords[i - 1], y_coords[i - 1])
        dist = math.hypot(x - x_prior, y - y_prior)
        velocity = dist / delay_s
        if velocity <= 0:
            continue

        feed = min(velocity * SPEED_MULTIPLIER, 11500)
        lines.append(f"G1 X{x:.3f} Y{y:.3f} F{feed:.3f}")

    return lines


def build_drawing_gcode(drawing_data) -> List[str]:
    strokes = drawing_data["drawing"]
    min_x = min(min(s[0]) for s in strokes)
    max_x = max(max(s[0]) for s in strokes)
    min_y = min(min(s[1]) for s in strokes)
    max_y = max(max(s[1]) for s in strokes)
    scaler = Augmenter(min_x, max_x, min_y, max_y, bounds=BOUNDS)

    lines = []
    for stroke in strokes:
        lines.extend(build_stroke_gcode(stroke, scaler))
    lines.append(f"G1 Z{PEN_UP_Z} F7000")
    return lines


# ------------------------- Dataset streaming --------------------------------

def fetch_ndjson_url(filename: str) -> str:
    return DATASET_URL + filename + ".ndjson"


def iter_drawings(url: str, limit: int = 1):
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        sys.exit("Error: The connection timed out. Please check your internet.")
    except requests.exceptions.RequestException as e:
        sys.exit(f"Error: A network error occurred: {e}")

    count = 0
    for line in response.iter_lines():
        if count >= limit:
            break
        if line:
            yield json.loads(line)
            count += 1


def main():
    position_log = PositionLog()

    try:
        controller = GRBLController(
            PORT, BAUD,
            position_log=position_log,
            status_hz=STATUS_POLL_HZ,
            only_log_pen_down=ONLY_SAMPLE_PEN_DOWN,
            mock=False,
        )
        controller.soft_reset_and_unlock()
    except serial.SerialException as e:
        print(f"Failed to open serial port:\n{e}\nFalling back to mock mode.")
        controller = GRBLController(
            PORT, BAUD,
            position_log=position_log,
            status_hz=0,  # no real status reports available in mock mode
            mock=True,
        )

    # Setup
    print("\nSetting up...")
    controller.send_line("G21")     # millimeters
    controller.send_line("G90")     # absolute positioning
    controller.send_line("$3=5")   # invert Y position negative -> positive
    #controller.send_line("$20=1")   # enable soft limits
    controller.send_line("$21=0")   # disable hard limits
    controller.send_line("$24=10000.0")   # homing locate feed rate
    controller.send_line("$25=10000.0")   # homing search seek rate
    controller.send_line("$32=0")   # disable laser mode
    controller.send_line(f"$130={X_MAX}")
    controller.send_line(f"$131={Y_MAX}")

    print("\nSettings:")
    controller.send_line("$$")

    # Home
    controller.home()
    controller.send_line("G10 L20 P1 X0 Y0 Z0")
    controller.wait_idle()

    controller.send_line("G0 X0 Y0")
    controller.send_line("G0 X100 Y-100")
    #controller.send_line("G0 X20 Y20")
    #controller.send_line("G0 X10 Y10")

    for word in WORDS_TO_PLOT:
        for drawing_data in iter_drawings(fetch_ndjson_url(word), limit=1):
            print(f"\n--- Plotting: {drawing_data.get('word', word)} ---")
            controller.stream(build_drawing_gcode(drawing_data))
            controller.wait_idle()

            print(f"\n--- Done: {drawing_data.get('word', word)} ---")

            # Re-home between drawings to minimize accumulated drift
            controller.home()

            position_log.save_csv("plot_position_log.csv")

    controller.send_line(f"G1 Z{PEN_UP_Z} F7000")
    controller.send_line("G0 X0 Y0")
    controller.wait_idle()

    position_log.save_csv("plot_position_log.csv")
    print(f"Logged {len(position_log.snapshot())} position samples to "
          f"plot_position_log.csv")

    controller.close()


if __name__ == "__main__":
    main()
