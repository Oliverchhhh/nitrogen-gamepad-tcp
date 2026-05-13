#!/usr/bin/env python3
"""
Mock inference server that sends a fixed gamepad action to the Windows client.

Runs on Linux, listens on TCP, receives frames from Recap (Windows),
and sends back a fixed gamepad action without running any model.

Usage:
    python mock_inference_server.py [--host 0.0.0.0] [--port 9001]
    python mock_inference_server.py --rx 1.0 --ry 0.0 --lx 0.0 --ly 0.0
"""

import argparse
import asyncio
import logging
import os
import sys

# Allow running from anywhere; resolve the elefant package relative to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elefant.data.proto import video_inference_pb2


class DecodedGamepadAction:
    """Mirrors the Python-side repr used by the real inference server."""

    def __init__(self, buttons_down, left_stick, right_stick, left_trigger, right_trigger):
        self.buttons_down = buttons_down
        self.left_stick = left_stick
        self.right_stick = right_stick
        self.left_trigger = left_trigger
        self.right_trigger = right_trigger

    def __repr__(self):
        return (
            f"DecodedGamepadAction("
            f"buttons_down={self.buttons_down}, "
            f"left_stick={self.left_stick}, "
            f"right_stick={self.right_stick}, "
            f"left_trigger={self.left_trigger}, "
            f"right_trigger={self.right_trigger})"
        )

    def to_keys(self):
        """Encode as legacy keys so the Windows client can parse it."""
        keys = [f"gamepad:{btn}" for btn in self.buttons_down]
        keys += [
            f"gamepad:lx={self.left_stick[0]:.4f}",
            f"gamepad:ly={self.left_stick[1]:.4f}",
            f"gamepad:rx={self.right_stick[0]:.4f}",
            f"gamepad:ry={self.right_stick[1]:.4f}",
            f"gamepad:lt={self.left_trigger:.4f}",
            f"gamepad:rt={self.right_trigger:.4f}",
        ]
        return keys


async def handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    fixed_action: DecodedGamepadAction,
) -> None:
    addr = writer.get_extra_info("peername")
    logging.info(f"New connection from {addr}")
    frame_i = 0
    try:
        while True:
            # Read 4-byte little-endian length prefix, then frame bytes
            length_bytes = await reader.readexactly(4)
            length = int.from_bytes(length_bytes, byteorder="little")
            frame_data = await reader.readexactly(length)
            frame = video_inference_pb2.Frame.FromString(frame_data)

            # Build Action with legacy gamepad keys
            action = video_inference_pb2.Action(
                keys=fixed_action.to_keys(),
                id=frame.id,
            )

            # Send 4-byte length prefix + serialized Action
            action_bytes = action.SerializeToString()
            writer.write(len(action_bytes).to_bytes(4, byteorder="little"))
            writer.write(action_bytes)
            await writer.drain()

            logging.info(f"Sending action: {fixed_action}")
            frame_i += 1

    except asyncio.IncompleteReadError:
        logging.info(f"Client {addr} disconnected after {frame_i} frames")
    except ConnectionError as e:
        logging.info(f"Connection error from {addr}: {e}")
    except Exception as e:
        logging.error(f"Error handling client {addr}: {e}")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def serve(host: str, port: int, fixed_action: DecodedGamepadAction) -> None:
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, fixed_action),
        host,
        port,
        limit=200_000,
    )
    logging.info(f"Mock inference server listening on {host}:{port}")
    logging.info(f"Will send fixed action: {fixed_action}")
    async with server:
        await server.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Mock gamepad inference server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9001, help="Bind port (default: 9001)")
    parser.add_argument("--lx", type=float, default=0.0, help="Left stick X (default: 0.0)")
    parser.add_argument("--ly", type=float, default=0.0, help="Left stick Y (default: 0.0)")
    parser.add_argument("--rx", type=float, default=1.0, help="Right stick X (default: 1.0)")
    parser.add_argument("--ry", type=float, default=0.0, help="Right stick Y (default: 0.0)")
    parser.add_argument("--lt", type=float, default=0.0, help="Left trigger (default: 0.0)")
    parser.add_argument("--rt", type=float, default=0.0, help="Right trigger (default: 0.0)")
    parser.add_argument(
        "--buttons", nargs="*", default=[], help="Buttons held down (e.g. south north)"
    )
    args = parser.parse_args()

    fixed_action = DecodedGamepadAction(
        buttons_down=args.buttons,
        left_stick=(args.lx, args.ly),
        right_stick=(args.rx, args.ry),
        left_trigger=args.lt,
        right_trigger=args.rt,
    )

    try:
        asyncio.run(serve(args.host, args.port, fixed_action))
    except KeyboardInterrupt:
        logging.info("Shutting down")


if __name__ == "__main__":
    main()
