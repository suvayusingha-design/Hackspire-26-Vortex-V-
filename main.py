"""
============================================================
GOD'S EYE: CROSS-PERSPECTIVE SURVIVOR LOCALIZATION & MR RESCUE
Central Launcher & System Orchestrator
============================================================
"""

import os
import sys
import socket
import threading
import time
import uvicorn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.server import app, init_subsystems, detection_worker, ctx


def get_local_ip() -> str:
    """Retrieve laptop's local Wi-Fi IP address for phone connection."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not send packets, resolves routing table interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def print_system_banner(host: str, port: int):
    local_ip = get_local_ip()
    phone_url = f"http://{local_ip}:{port}"
    debug_url = f"http://{local_ip}:{port}/debug"

    print("\n" + "=" * 50)
    print("                GOD'S EYE                ")
    print("      LIVE RESCUE ASSISTANCE SYSTEM      ")
    print("=" * 50)
    
    # Check status of components
    cam_status = "CONNECTED" if ctx["camera_stream"].is_connected else "CONNECTING..."
    yolo_status = "RUNNING" if ctx["yolo_adapter"] is not None else "ERROR"
    mmwave_status = "CONNECTED" if ctx["mmwave_adapter"].connected else "WAITING FOR DATA"
    mode = ctx["config"].get("mode", {}).get("mode", "LIVE").upper()

    print(f"System Mode : {mode}")
    print(f"Drone Camera: {cam_status}")
    print(f"YOLO11      : {yolo_status}")
    print(f"C4001 mmWave: {mmwave_status}")
    print(f"SensorFusion: ACTIVE")
    print(f"AR Renderer : ACTIVE")
    print("-" * 50)
    print(f"RESCUER PHONE VR URL : {phone_url}")
    print(f"DEBUG & CALIBRATION   : {debug_url}")
    print("=" * 50 + "\n")


def main():
    config_file = os.path.join(PROJECT_ROOT, "config", "config.yaml")

    print("[BOOT] Initializing GOD'S EYE subsystems...")
    init_subsystems(config_file)

    # Launch YOLO detection background worker
    worker_thread = threading.Thread(target=detection_worker, daemon=True)
    worker_thread.start()

    cfg = ctx["config"].get("server", {})
    host = cfg.get("host", "0.0.0.0")
    port = cfg.get("port", 8000)

    # Print system startup banner after a brief pause for camera handshake
    threading.Timer(1.5, lambda: print_system_banner(host, port)).start()

    # Start FastAPI / Uvicorn server
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
