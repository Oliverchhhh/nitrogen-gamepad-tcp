#!/usr/bin/env python3
"""Analyze converted annotation.proto file."""

import sys
from pathlib import Path
from elefant.data.proto import video_annotation_pb2

if len(sys.argv) < 2:
    print("Usage: python analyze_proto.py <path_to_annotation.proto>")
    sys.exit(1)

proto_path = Path(sys.argv[1])
if not proto_path.exists():
    print(f"File not found: {proto_path}")
    sys.exit(1)

# Load proto
va = video_annotation_pb2.VideoAnnotation()
va.ParseFromString(proto_path.read_bytes())

print("=" * 80)
print("ANNOTATION.PROTO ANALYSIS")
print("=" * 80)

# Metadata
print("\n📋 METADATA:")
print(f"  ID: {va.metadata.id}")
print(f"  FPS: {va.metadata.frames_per_second}")
if va.metadata.HasField('env'):
    print(f"  Environment: {va.metadata.env.env}")
    print(f"  Env Subtype: {va.metadata.env.env_subtype}")
if va.metadata.HasField('video_source_info'):
    if va.metadata.video_source_info.source.HasField('youtube_source'):
        print(f"  YouTube Video ID: {va.metadata.video_source_info.source.youtube_source.video_id}")

# Frame annotations
print(f"\n🎬 FRAME ANNOTATIONS:")
print(f"  Total frames: {len(va.frame_annotations)}")

# Analyze first few frames
print(f"\n📊 FIRST 5 FRAMES:")
for i in range(min(5, len(va.frame_annotations))):
    frame = va.frame_annotations[i]
    print(f"\n  Frame {i}:")
    print(f"    Frame time: {frame.frame_time} microseconds")
    
    if frame.user_action.is_known:
        if frame.user_action.HasField('game_pad'):
            gp = frame.user_action.game_pad
            
            # Buttons
            buttons_pressed = []
            if gp.buttons.south:
                buttons_pressed.append("south")
            if gp.buttons.north:
                buttons_pressed.append("north")
            if gp.buttons.east:
                buttons_pressed.append("east")
            if gp.buttons.west:
                buttons_pressed.append("west")
            if gp.buttons.dpad_up:
                buttons_pressed.append("dpad_up")
            if gp.buttons.dpad_down:
                buttons_pressed.append("dpad_down")
            if gp.buttons.dpad_left:
                buttons_pressed.append("dpad_left")
            if gp.buttons.dpad_right:
                buttons_pressed.append("dpad_right")
            if gp.buttons.start:
                buttons_pressed.append("start")
            if gp.buttons.select:
                buttons_pressed.append("select")
            if gp.buttons.left_bumper:
                buttons_pressed.append("left_bumper")
            if gp.buttons.right_bumper:
                buttons_pressed.append("right_bumper")
            
            print(f"    Buttons pressed: {buttons_pressed if buttons_pressed else 'none'}")
            
            # Sticks
            print(f"    Left stick: ({gp.left_stick.x:.3f}, {gp.left_stick.y:.3f}), pressed={gp.left_stick.pressed}")
            print(f"    Right stick: ({gp.right_stick.x:.3f}, {gp.right_stick.y:.3f}), pressed={gp.right_stick.pressed}")
            
            # Triggers
            print(f"    Triggers: L={gp.left_trigger:.3f}, R={gp.right_trigger:.3f}")
        else:
            print(f"    No gamepad action")
    else:
        print(f"    User action: unknown")

# Statistics
print(f"\n📈 STATISTICS:")
total_frames = len(va.frame_annotations)
frames_with_actions = sum(1 for f in va.frame_annotations if f.user_action.is_known)
frames_with_gamepad = sum(1 for f in va.frame_annotations 
                          if f.user_action.is_known and f.user_action.HasField('game_pad'))

print(f"  Total frames: {total_frames}")
print(f"  Frames with known actions: {frames_with_actions} ({100*frames_with_actions/total_frames:.1f}%)")
print(f"  Frames with gamepad actions: {frames_with_gamepad} ({100*frames_with_gamepad/total_frames:.1f}%)")

# Count button presses
button_counts = {
    'south': 0, 'north': 0, 'east': 0, 'west': 0,
    'dpad_up': 0, 'dpad_down': 0, 'dpad_left': 0, 'dpad_right': 0,
    'start': 0, 'select': 0, 'left_bumper': 0, 'right_bumper': 0,
}

for frame in va.frame_annotations:
    if frame.user_action.is_known and frame.user_action.HasField('game_pad'):
        gp = frame.user_action.game_pad
        if gp.buttons.south: button_counts['south'] += 1
        if gp.buttons.north: button_counts['north'] += 1
        if gp.buttons.east: button_counts['east'] += 1
        if gp.buttons.west: button_counts['west'] += 1
        if gp.buttons.dpad_up: button_counts['dpad_up'] += 1
        if gp.buttons.dpad_down: button_counts['dpad_down'] += 1
        if gp.buttons.dpad_left: button_counts['dpad_left'] += 1
        if gp.buttons.dpad_right: button_counts['dpad_right'] += 1
        if gp.buttons.start: button_counts['start'] += 1
        if gp.buttons.select: button_counts['select'] += 1
        if gp.buttons.left_bumper: button_counts['left_bumper'] += 1
        if gp.buttons.right_bumper: button_counts['right_bumper'] += 1

print(f"\n🎮 BUTTON PRESS COUNTS:")
for button, count in sorted(button_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {button}: {count} frames ({100*count/total_frames:.1f}%)")

# Stick movement statistics
left_stick_moved = sum(1 for f in va.frame_annotations 
                      if f.user_action.is_known and f.user_action.HasField('game_pad') and
                      (abs(f.user_action.game_pad.left_stick.x) > 0.01 or 
                       abs(f.user_action.game_pad.left_stick.y) > 0.01))
right_stick_moved = sum(1 for f in va.frame_annotations 
                       if f.user_action.is_known and f.user_action.HasField('game_pad') and
                       (abs(f.user_action.game_pad.right_stick.x) > 0.01 or 
                        abs(f.user_action.game_pad.right_stick.y) > 0.01))

print(f"\n🕹️ STICK MOVEMENT:")
print(f"  Left stick moved: {left_stick_moved} frames ({100*left_stick_moved/total_frames:.1f}%)")
print(f"  Right stick moved: {right_stick_moved} frames ({100*right_stick_moved/total_frames:.1f}%)")

# Trigger usage
left_trigger_used = sum(1 for f in va.frame_annotations 
                       if f.user_action.is_known and f.user_action.HasField('game_pad') and
                       f.user_action.game_pad.left_trigger > 0.01)
right_trigger_used = sum(1 for f in va.frame_annotations 
                        if f.user_action.is_known and f.user_action.HasField('game_pad') and
                        f.user_action.game_pad.right_trigger > 0.01)

print(f"\n🔫 TRIGGER USAGE:")
print(f"  Left trigger used: {left_trigger_used} frames ({100*left_trigger_used/total_frames:.1f}%)")
print(f"  Right trigger used: {right_trigger_used} frames ({100*right_trigger_used/total_frames:.1f}%)")

print("\n" + "=" * 80)
