import os

main_block = """
def start_engine() -> None:
    # ── Set Process Priority to HIGH ──
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10)
    except Exception as e:
        log.warning("Could not set process priority: %s", e)

    shared_state = GestureState()
    stop_event = threading.Event()

    server = _WebSocketServer(shared_state)
    server.start()

    log.info("Starting engine...")
    try:
        if _USE_TASKS_API:
            _camera_loop_tasks(shared_state, stop_event)
        else:
            _camera_loop_legacy(shared_state, stop_event)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received. Shutting down gracefully...")
    finally:
        stop_event.set()
        server.stop()
        log.info("Engine shutdown complete.")

if __name__ == "__main__":
    start_engine()
"""

with open('c:/Users/kito/Desktop/TestCV/GrandII/gesture_engine.py', 'a', encoding='utf-8') as f:
    f.write(main_block)

print("Restored start_engine() and __main__ block.")
