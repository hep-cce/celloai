#!/usr/bin/env python3
import asyncio
import websockets
import subprocess
import signal
import sys
import json

HOST = "0.0.0.0"
PORT = 8765

def start_subprocess():
    """
    Launch a Python subprocess defining our function(s).
    Instead of a separate .py script, we run inline code via `-c`.
    """
    code = r'''
print("Enter a query")
while True:
    query = input(" ")
    if query == "exit":
        break
    # Print the result
    print("> Question:")
    print(query)
'''
    # Spawn the subprocess
    return subprocess.Popen(
        ["python", "./scripts/run_chatbot.py"],
        #[sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,   # read/write text (not bytes)
        bufsize=1    # line-buffered
    )

async def script_handler(websocket):
    """
    1. Start the custom inline-subprocess via start_subprocess().
    2. Forward subprocess stdout to the WebSocket.
    3. Forward WebSocket messages to the subprocess stdin.
    """
    process = start_subprocess()

    async def read_subprocess_output():
        loop = asyncio.get_running_loop()
        while True:
            # Read one line at a time from subprocess stdout
            line = await loop.run_in_executor(None, process.stdout.readline)
            if not line:
                break  # Subprocess ended or no more output
            await websocket.send(line)

        await websocket.close()

    # Background task to capture the subprocess output
    read_task = asyncio.create_task(read_subprocess_output())

    try:
        # Relay WebSocket input -> subprocess stdin
        async for message in websocket:
            if process.stdin:
                process.stdin.write(message + "\n")
                process.stdin.flush()
    except websockets.ConnectionClosed:
        pass
    finally:
        # Terminate the subprocess if still running
        read_task.cancel()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

async def main():
    async with websockets.serve(script_handler, HOST, PORT):
        print(f"Server running at ws://{HOST}:{PORT}")
        await asyncio.Future()  # Keep running

if __name__ == "__main__":
    asyncio.run(main())

