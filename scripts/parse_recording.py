#!/usr/bin/env python3
"""Parse a recap recording annotation.proto and print summary."""
import sys
sys.path.insert(0, '.')
from elefant.data.proto import video_annotation_pb2

import os, platform
# _WIN_PATH = r"C:\Users\1897\AppData\Local\Temp\com.elefant.recap\recordings\019d3d92-4289-7132-ac94-49a84fb1658a\annotation.proto"
# _WSL_PATH = "/mnt/c/Users/1897/AppData/Local/Temp/com.elefant.recap/recordings/019d3d92-4289-7132-ac94-49a84fb1658a/annotation.proto"
_WIN_PATH = r"C:\Users\1897\AppData\Local\Temp\com.elefant.recap\recordings\019d3dbe-7698-7fc1-b963-fe496b6736da\annotation.proto"
_WSL_PATH = "/mnt/c/Users/1897/AppData/Local/Temp/com.elefant.recap/recordings/019d3dbe-7698-7fc1-b963-fe496b6736da/annotation.proto"
RECORDING = _WIN_PATH if platform.system() == "Windows" else _WSL_PATH

data = open(RECORDING, 'rb').read()
ann = video_annotation_pb2.VideoAnnotation()
ann.ParseFromString(data)

meta = ann.metadata
print(f"ID: {meta.id}")
print(f"FPS: {meta.frames_per_second}")
print(f"Total frames: {len(ann.frame_annotations)}")
print(f"Version: {ann.version}")
print()

for i in [0, 1, 2, -3, -2, -1]:
    if abs(i) > len(ann.frame_annotations):
        continue
    fa = ann.frame_annotations[i]
    ua = fa.user_action
    sa = fa.system_action
    print(f"--- Frame {i} ---")
    print(f"  user keys: {list(ua.keyboard.keys)}")
    print(f"  user mouse buttons: {list(ua.mouse.buttons_down)}")
    has_user_gp = ua.HasField("game_pad")
    print(f"  user gamepad present: {has_user_gp}")
    if has_user_gp:
        gp = ua.game_pad
        b = gp.buttons
        print(f"    buttons: S={b.south} N={b.north} E={b.east} W={b.west} start={b.start} select={b.select}")
        print(f"    dpad: U={b.dpad_up} D={b.dpad_down} L={b.dpad_left} R={b.dpad_right}")
        print(f"    bumpers: LB={b.left_bumper} RB={b.right_bumper} LT_btn={b.left_thumb} RT_btn={b.right_thumb}")
        print(f"    left_stick: ({gp.left_stick.x}, {gp.left_stick.y})")
        print(f"    right_stick: ({gp.right_stick.x}, {gp.right_stick.y})")
        print(f"    triggers: L={gp.left_trigger} R={gp.right_trigger}")

    print(f"  system keys: {list(sa.keyboard.keys)}")
    has_sys_gp = sa.HasField("game_pad")
    print(f"  system gamepad present: {has_sys_gp}")
    if has_sys_gp:
        gp = sa.game_pad
        b = gp.buttons
        print(f"    buttons: S={b.south} N={b.north} E={b.east} W={b.west}")
        print(f"    left_stick: ({gp.left_stick.x}, {gp.left_stick.y})")
        print(f"    right_stick: ({gp.right_stick.x}, {gp.right_stick.y})")
        print(f"    triggers: L={gp.left_trigger} R={gp.right_trigger}")

    print(f"  input_events: {len(fa.input_events)}")
    print()

# Summary: how many frames have inference_running
try:
    inf_count = sum(1 for fa in ann.frame_annotations if fa.inference_running)
    print(f"Frames with inference_running=True: {inf_count}/{len(ann.frame_annotations)}")
except AttributeError:
    print("(inference_running field not present in this proto version)")

# Summary: how many frames have system gamepad
sys_gp_count = sum(1 for fa in ann.frame_annotations if fa.system_action.HasField("game_pad"))
print(f"Frames with system gamepad action: {sys_gp_count}/{len(ann.frame_annotations)}")

# Summary: how many frames have user gamepad
usr_gp_count = sum(1 for fa in ann.frame_annotations if fa.user_action.HasField("game_pad"))
print(f"Frames with user gamepad action: {usr_gp_count}/{len(ann.frame_annotations)}")

# Check if ANY system action has any content at all
sys_any = 0
for fa in ann.frame_annotations:
    sa = fa.system_action
    if list(sa.keyboard.keys) or list(sa.mouse.buttons_down) or sa.HasField("game_pad"):
        sys_any += 1
print(f"Frames with ANY system action: {sys_any}/{len(ann.frame_annotations)}")

# Check all available fields on frame_annotations[0]
print(f"\nAvailable fields on FrameAnnotation: {[f.name for f in ann.frame_annotations[0].DESCRIPTOR.fields]}")