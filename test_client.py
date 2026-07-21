"""
test_client.py
==============
Simple WebSocket test client — prints incoming gesture data to the console.
Use this to verify the gesture engine is running and streaming correctly.

Usage:
    python test_client.py
"""

import asyncio
import json
import websockets


WS_URI = "ws://localhost:8765"


async def listen() -> None:
    print(f"Connecting to {WS_URI} …")
    try:
        async with websockets.connect(WS_URI) as ws:
            print("Connected. Receiving gesture data (Ctrl+C to quit):\n")
            prev_state = None
            prev_gesture = None
            async for raw in ws:
                data = json.loads(raw)
                state   = data["state"]
                gesture = data["gesture"]

                # Only print when something interesting changes (reduces console spam)
                if state != prev_state or gesture != prev_gesture:
                    print(
                        f"  x={data['x']:>4}  y={data['y']:>4}"
                        f"  state={state:<6}  gesture={gesture}"
                    )
                    prev_state   = state
                    prev_gesture = gesture

    except ConnectionRefusedError:
        print(f"ERROR: Could not connect to {WS_URI}. Is gesture_engine.py running?")
    except KeyboardInterrupt:
        print("\nDisconnected.")


if __name__ == "__main__":
    asyncio.run(listen())
