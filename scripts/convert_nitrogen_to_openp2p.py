#!/usr/bin/env python3
"""
Convert NitroGen dataset format to open-p2p format.

This script converts NitroGen chunks (with controller actions) to open-p2p format,
preserving the controller action space (not converting to keyboard+mouse).

Key points:
- NitroGen uses controller/gamepad action space (PS4/Xbox controller)
- open-p2p uses keyboard+mouse action space
- We preserve NitroGen's controller actions in GamePadAction fields
- We create open-p2p compatible directory structure and annotation.proto files
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from elefant.data.proto import video_annotation_pb2, shared_pb2


def load_nitrogen_metadata(metadata_path: Path) -> dict:
    """Load NitroGen metadata.json file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_nitrogen_actions(actions_path: Path) -> pd.DataFrame:
    """Load NitroGen actions from parquet file."""
    return pd.read_parquet(actions_path)


def find_column_by_pattern(columns: list, patterns: list) -> Optional[str]:
    """Find a column name matching any of the given patterns."""
    for col in columns:
        col_lower = col.lower()
        for pattern in patterns:
            if pattern.lower() in col_lower:
                return col
    return None


def create_gamepad_action_from_nitrogen(
    row: pd.Series,
    columns: list,
) -> video_annotation_pb2.GamePadAction:
    """
    Create a GamePadAction from NitroGen action row.
    
    Based on actual NitroGen parquet structure:
    - Buttons: south, north, east, west, dpad_*, start, back, guide, left_shoulder, right_shoulder, left_thumb, right_thumb
    - Triggers: left_trigger, right_trigger (int32, likely 0-255 range)
    - Sticks: j_left, j_right (numpy arrays [x, y])
    """
    gamepad_action = video_annotation_pb2.GamePadAction()
    
    # Create buttons
    buttons = video_annotation_pb2.GamePadButtons()
    
    # Direct button mappings (NitroGen uses exact names matching GamePadButtons)
    button_mappings = {
        'south': 'south',           # PS4: X, Xbox: A
        'north': 'north',           # PS4: Triangle, Xbox: Y
        'west': 'west',             # PS4: Square, Xbox: X
        'east': 'east',             # PS4: Circle, Xbox: B
        'dpad_up': 'dpad_up',
        'dpad_down': 'dpad_down',
        'dpad_left': 'dpad_left',
        'dpad_right': 'dpad_right',
        'start': 'start',
        'back': 'select',           # NitroGen's 'back' maps to 'select' in GamePadButtons
        'left_shoulder': 'left_bumper',   # NitroGen's 'left_shoulder' = L1/LB
        'right_shoulder': 'right_bumper', # NitroGen's 'right_shoulder' = R1/RB
    }
    
    for nitrogen_col, button_field in button_mappings.items():
        if nitrogen_col in row.index:
            value = row[nitrogen_col]
            # Handle int32 (0 or 1) or boolean values
            if isinstance(value, (bool, int, float, np.integer)) and value:
                setattr(buttons, button_field, True)
    
    # Note: 'guide' button exists in NitroGen but not in GamePadButtons, skip it
    # Note: 'left_thumb' and 'right_thumb' are handled as stick.pressed below
    
    gamepad_action.buttons.CopyFrom(buttons)
    
    # Create left stick from j_left array [x, y]
    left_stick = video_annotation_pb2.Stick()
    if 'j_left' in row.index:
        j_left = row['j_left']
        if isinstance(j_left, (list, tuple, np.ndarray)):
            # Handle numpy array or list [x, y]
            j_left_array = np.array(j_left)
            if len(j_left_array) >= 2:
                left_stick.x = float(j_left_array[0])
                left_stick.y = float(j_left_array[1])
        else:
            left_stick.x = 0.0
            left_stick.y = 0.0
    else:
        left_stick.x = 0.0
        left_stick.y = 0.0
    
    # left_thumb indicates if left stick is pressed
    if 'left_thumb' in row.index:
        left_stick.pressed = bool(row['left_thumb'])
    else:
        left_stick.pressed = False
    
    gamepad_action.left_stick.CopyFrom(left_stick)
    
    # Create right stick from j_right array [x, y]
    right_stick = video_annotation_pb2.Stick()
    if 'j_right' in row.index:
        j_right = row['j_right']
        if isinstance(j_right, (list, tuple, np.ndarray)):
            # Handle numpy array or list [x, y]
            j_right_array = np.array(j_right)
            if len(j_right_array) >= 2:
                right_stick.x = float(j_right_array[0])
                right_stick.y = float(j_right_array[1])
        else:
            right_stick.x = 0.0
            right_stick.y = 0.0
    else:
        right_stick.x = 0.0
        right_stick.y = 0.0
    
    # right_thumb indicates if right stick is pressed
    if 'right_thumb' in row.index:
        right_stick.pressed = bool(row['right_thumb'])
    else:
        right_stick.pressed = False
    
    gamepad_action.right_stick.CopyFrom(right_stick)
    
    # Triggers (int32, likely 0-255 range, normalize to 0.0-1.0)
    if 'left_trigger' in row.index:
        trigger_val = row['left_trigger']
        # Normalize if it's in 0-255 range, otherwise assume 0.0-1.0
        if isinstance(trigger_val, (int, np.integer)) and trigger_val > 1:
            gamepad_action.left_trigger = float(trigger_val) / 255.0
        else:
            gamepad_action.left_trigger = float(trigger_val)
    else:
        gamepad_action.left_trigger = 0.0
    
    if 'right_trigger' in row.index:
        trigger_val = row['right_trigger']
        # Normalize if it's in 0-255 range, otherwise assume 0.0-1.0
        if isinstance(trigger_val, (int, np.integer)) and trigger_val > 1:
            gamepad_action.right_trigger = float(trigger_val) / 255.0
        else:
            gamepad_action.right_trigger = float(trigger_val)
    else:
        gamepad_action.right_trigger = 0.0
    
    return gamepad_action


def create_video_annotation(
    nitrogen_metadata: dict,
    actions_df: pd.DataFrame,
    sample_uuid: str,
) -> video_annotation_pb2.VideoAnnotation:
    """
    Create a VideoAnnotation proto from NitroGen data.
    
    Args:
        nitrogen_metadata: Metadata from NitroGen metadata.json
        actions_df: DataFrame with actions from actions_processed.parquet
        sample_uuid: UUID for this sample
    
    Returns:
        VideoAnnotation proto object
    """
    video_annotation = video_annotation_pb2.VideoAnnotation()
    
    # Set metadata
    metadata = video_annotation_pb2.VideoAnnotationMetadata()
    metadata.id = sample_uuid
    metadata.frames_per_second = 60.0  # NitroGen typically uses 60 FPS
    
    # Set environment info
    env = video_annotation_pb2.VideoAnnotationEnv()
    env.env = nitrogen_metadata.get('game', 'nitrogen_unknown')
    env.env_subtype = nitrogen_metadata.get('controller_type', 'unknown')
    metadata.env.CopyFrom(env)
    
    # Set video source info (YouTube)
    video_source_info = video_annotation_pb2.VideoSourceInfo()
    source = video_annotation_pb2.Source()
    youtube_source = video_annotation_pb2.YoutubeSource()
    youtube_source.video_id = nitrogen_metadata['original_video']['video_id']
    source.youtube_source.CopyFrom(youtube_source)
    video_source_info.source.CopyFrom(source)
    metadata.video_source_info.CopyFrom(video_source_info)
    
    video_annotation.metadata.CopyFrom(metadata)
    
    # Create frame annotations
    n_frames = len(actions_df)
    for i, (_, row) in enumerate(actions_df.iterrows()):
        frame_annotation = video_annotation_pb2.FrameAnnotation()
        
        # Create LowLevelAction with GamePadAction
        low_level_action = video_annotation_pb2.LowLevelAction()
        low_level_action.is_known = True
        
        # Create GamePadAction from NitroGen row
        gamepad_action = create_gamepad_action_from_nitrogen(row, actions_df.columns.tolist())
        low_level_action.game_pad.CopyFrom(gamepad_action)
        
        frame_annotation.user_action.CopyFrom(low_level_action)
        
        # Set frame time (in microseconds)
        # Assuming 60 FPS, each frame is ~16666 microseconds
        frame_annotation.frame_time = int(i * 16666)
        
        video_annotation.frame_annotations.append(frame_annotation)
    
    return video_annotation


def convert_nitrogen_chunk(
    nitrogen_chunk_dir: Path,
    output_base_dir: Path,
    sample_uuid: Optional[str] = None,
) -> Path:
    """
    Convert a single NitroGen chunk to open-p2p format.
    
    Args:
        nitrogen_chunk_dir: Path to NitroGen chunk directory (e.g., .../_5o1qSXWFfA_chunk_0291)
        output_base_dir: Base directory for output (e.g., dataset/)
        sample_uuid: Optional UUID for the sample (generated if not provided)
    
    Returns:
        Path to the created output directory
    """
    # Generate UUID if not provided
    if sample_uuid is None:
        sample_uuid = str(uuid.uuid4())
    
    # Create output directory
    output_dir = output_base_dir / sample_uuid
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load NitroGen data
    metadata_path = nitrogen_chunk_dir / 'metadata.json'
    actions_path = nitrogen_chunk_dir / 'actions_processed.parquet'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_path}")
    
    nitrogen_metadata = load_nitrogen_metadata(metadata_path)
    actions_df = load_nitrogen_actions(actions_path)
    
    print(f"Converting chunk: {nitrogen_chunk_dir.name}")
    print(f"  Video ID: {nitrogen_metadata['original_video']['video_id']}")
    print(f"  Chunk ID: {nitrogen_metadata['chunk_id']}")
    print(f"  Frames: {len(actions_df)}")
    print(f"  Actions columns: {actions_df.columns.tolist()}")
    
    # Create VideoAnnotation proto
    video_annotation = create_video_annotation(
        nitrogen_metadata,
        actions_df,
        sample_uuid,
    )
    
    # Write annotation.proto
    annotation_path = output_dir / 'annotation.proto'
    with open(annotation_path, 'wb') as f:
        f.write(video_annotation.SerializeToString())
    
    print(f"  Created: {annotation_path}")
    print(f"  Output directory: {output_dir}")
    
    # Note: Video files need to be downloaded separately from NitroGen video dataset
    # For now, we only create the annotation.proto file
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert NitroGen dataset chunks to open-p2p format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input NitroGen chunk directory or parent directory containing chunks',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dataset',
        help='Output base directory (default: dataset)',
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively find all chunks in input directory',
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_base_dir = Path(args.output)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    if args.recursive:
        # Find all chunk directories (containing metadata.json and actions_processed.parquet)
        chunk_dirs = []
        for metadata_file in input_path.rglob('metadata.json'):
            chunk_dir = metadata_file.parent
            if (chunk_dir / 'actions_processed.parquet').exists():
                chunk_dirs.append(chunk_dir)
        
        print(f"Found {len(chunk_dirs)} chunks to convert")
        for chunk_dir in chunk_dirs:
            try:
                convert_nitrogen_chunk(chunk_dir, output_base_dir)
            except Exception as e:
                print(f"Error converting {chunk_dir}: {e}")
                continue
    else:
        # Single chunk directory
        convert_nitrogen_chunk(input_path, output_base_dir)
    
    print(f"\nConversion complete! Output directory: {output_base_dir}")


if __name__ == '__main__':
    main()
