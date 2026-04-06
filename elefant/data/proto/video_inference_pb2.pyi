import shared_pb2 as _shared_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Frame(_message.Message):
    __slots__ = ("width", "height", "data", "id")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    data: bytes
    id: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., data: _Optional[bytes] = ..., id: _Optional[int] = ...) -> None: ...

class MouseAction(_message.Message):
    __slots__ = ("mouse_delta_px", "mouse_pos", "scroll_delta_px", "buttons_down")
    MOUSE_DELTA_PX_FIELD_NUMBER: _ClassVar[int]
    MOUSE_POS_FIELD_NUMBER: _ClassVar[int]
    SCROLL_DELTA_PX_FIELD_NUMBER: _ClassVar[int]
    BUTTONS_DOWN_FIELD_NUMBER: _ClassVar[int]
    mouse_delta_px: _shared_pb2.Vec2Int
    mouse_pos: _shared_pb2.Vec2Float
    scroll_delta_px: _shared_pb2.Vec2Int
    buttons_down: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, mouse_delta_px: _Optional[_Union[_shared_pb2.Vec2Int, _Mapping]] = ..., mouse_pos: _Optional[_Union[_shared_pb2.Vec2Float, _Mapping]] = ..., scroll_delta_px: _Optional[_Union[_shared_pb2.Vec2Int, _Mapping]] = ..., buttons_down: _Optional[_Iterable[str]] = ...) -> None: ...

class GamepadAction(_message.Message):
    __slots__ = ("buttons_down", "left_stick", "right_stick", "left_trigger", "right_trigger")
    BUTTONS_DOWN_FIELD_NUMBER: _ClassVar[int]
    LEFT_STICK_FIELD_NUMBER: _ClassVar[int]
    RIGHT_STICK_FIELD_NUMBER: _ClassVar[int]
    LEFT_TRIGGER_FIELD_NUMBER: _ClassVar[int]
    RIGHT_TRIGGER_FIELD_NUMBER: _ClassVar[int]
    buttons_down: _containers.RepeatedScalarFieldContainer[str]
    left_stick: _shared_pb2.Vec2Float
    right_stick: _shared_pb2.Vec2Float
    left_trigger: float
    right_trigger: float
    def __init__(self, buttons_down: _Optional[_Iterable[str]] = ..., left_stick: _Optional[_Union[_shared_pb2.Vec2Float, _Mapping]] = ..., right_stick: _Optional[_Union[_shared_pb2.Vec2Float, _Mapping]] = ..., left_trigger: _Optional[float] = ..., right_trigger: _Optional[float] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("keys", "id", "mouse_action", "gamepad_action")
    KEYS_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    MOUSE_ACTION_FIELD_NUMBER: _ClassVar[int]
    GAMEPAD_ACTION_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedScalarFieldContainer[str]
    id: int
    mouse_action: MouseAction
    gamepad_action: GamepadAction
    def __init__(self, keys: _Optional[_Iterable[str]] = ..., id: _Optional[int] = ..., mouse_action: _Optional[_Union[MouseAction, _Mapping]] = ..., gamepad_action: _Optional[_Union[GamepadAction, _Mapping]] = ...) -> None: ...
