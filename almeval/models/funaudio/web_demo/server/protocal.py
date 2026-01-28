# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
server protocal
"""

class MessageType:
    HANDSHAKE = 0x00
    AUDIO = 0x01
    TEXT = 0x02
    CONTROL = 0x03
    METADATA = 0x04
    ERROR = 0x05
    PING = 0x06
    COLOREDTEXT = 0x07


class ControlMessage:
    START = 0x00
    END_TURN = 0x01
    PAUSE = 0x02
    RESTART = 0x03


def encode_handshake(version: int = 0, model: int = 0) -> bytes:
    return bytes([MessageType.HANDSHAKE, version, model])


def decode_message(data: bytes) -> dict:
    if len(data) == 0:
        raise ValueError("Empty message data")
    
    msg_type = data[0]
    payload = data[1:]
    
    if msg_type == MessageType.HANDSHAKE:
        version = payload[0] if len(payload) > 0 else 0
        model = payload[1:] if len(payload) > 1 else 0
        return {
            'type': 'handshake',
            'version': version,
            'model': model
        }
    
    elif msg_type == MessageType.AUDIO:
        return {
            'type': 'audio',
            'data': payload
        }
    
    elif msg_type == MessageType.TEXT:
        return {
            'type': 'text',
            'data': payload.decode('utf-8')
        }
    
    elif msg_type == MessageType.COLOREDTEXT:
        color = payload[0] if len(payload) > 0 else 0
        text = payload[1:].decode('utf-8') if len(payload) > 1 else ""
        return {
            'type': 'coloredtext',
            'color': color,
            'data': text
        }
    
    elif msg_type == MessageType.CONTROL:
        action = payload[0] if len(payload) > 0 else 0
        action_names = {
            ControlMessage.START: 'start',
            ControlMessage.END_TURN: 'endTurn',
            ControlMessage.PAUSE: 'pause',
            ControlMessage.RESTART: 'restart'
        }
        return {
            'type': 'control',
            'action': action_names.get(action, 'unknown')
        }
    
    elif msg_type == MessageType.METADATA:
        import json
        metadata = json.loads(payload.decode('utf-8'))
        return {
            'type': 'metadata',
            'data': metadata
        }
    
    elif msg_type == MessageType.ERROR:
        return {
            'type': 'error',
            'data': payload.decode('utf-8')
        }
    
    elif msg_type == MessageType.PING:
        return {
            'type': 'ping'
        }
    
    else:
        raise ValueError(f"Unknown message type: 0x{msg_type:02x}")

