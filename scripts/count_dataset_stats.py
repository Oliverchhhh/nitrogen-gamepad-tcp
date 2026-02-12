from pathlib import Path
from elefant.data.proto import video_annotation_pb2

# 直接指定这个样本的目录
sample_dir = Path("dataset/0195bfe5-a84c-71d1-9071-9e42b83ab6a3")
proto_path = sample_dir / "annotation.proto"

va = video_annotation_pb2.VideoAnnotation()
va.ParseFromString(proto_path.read_bytes())

n_frames = len(va.frame_annotations)
n_actions = sum(
    1 for fa in va.frame_annotations
    if fa.user_action.is_known or fa.system_action.is_known
)

print(sample_dir.name, n_frames, n_actions)