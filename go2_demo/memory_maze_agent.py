"""Single-call model client for the Memory Maze adapter, without Go2 dependencies.

The caller owns observation filtering, argument validation and tool execution.
Only complete public request/result turns enter the retained dialogue.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json

import numpy as np
from PIL import Image

from .models import APIModel, MAX_IMAGE_CHARS, ModelError, ModelProtocolError, _json_text, _usage_counters
from .types import ToolCall


def tool_schema(name, description, properties, required=None):
    """Build a function schema; omitted ``required`` means all supplied fields."""
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": deepcopy(properties),
                       "required": list(properties) if required is None else list(required),
                       "additionalProperties": False},
    }}


def _png_url(rgb):
    if rgb is None:
        return None
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or not rgb.shape[0] or not rgb.shape[1]):
        raise ModelProtocolError("The public image must be a nonempty uint8 RGB array.")
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    value = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    if len(value) > MAX_IMAGE_CHARS:
        raise ModelProtocolError("The public image exceeded the allowed size.")
    return value


class MazeClient(APIModel):
    """Exactly one paid request per choose; optional complete public dialogue."""

    def __init__(self, *args, system_prompt: str, retain_context: bool, **kwargs):
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ModelError("A nonempty system prompt is required.", "MODEL_CONFIG_ERROR")
        if type(retain_context) is not bool:
            raise ModelError("retain_context must be a boolean.", "MODEL_CONFIG_ERROR")
        super().__init__(*args, **kwargs)
        self.extra_body.update(thinking={"type": "disabled"}, temperature=0)
        self.requires_reasoning_content = False
        self.system_prompt = system_prompt
        self.retain_context = retain_context
        self.raw_usage = None
        self.response_finish_reason = None
        self.last_decision = None
        self._dialogue = []
        self._turns = 0
        self._pending_user = None
        self._pending_decision = None

    def build_messages(self, observation, image_data_url, history):
        content = _json_text(observation["input"])
        if image_data_url is not None:
            content = [{"type": "text", "text": content},
                       {"type": "image_url", "image_url": {"url": image_data_url}}]
        self._pending_user = {"role": "user", "content": content}
        return [{"role": "system", "content": self.system_prompt},
                *deepcopy(self._dialogue), deepcopy(self._pending_user)]

    async def _request(self, payload, timeout_s):
        body = await super()._request(
            {**payload, "tool_choice": "required", "parallel_tool_calls": False}, timeout_s)
        # Keep usage even when APIModel subsequently rejects the tool protocol.
        try:
            response = json.loads(body)
            if isinstance(response, dict):
                self.raw_usage = _usage_counters(response.get("usage"))
                choices = response.get("choices")
                if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
                    reason = choices[0].get("finish_reason")
                    if reason in {"length", "tool_calls", "stop", "content_filter"}:
                        self.response_finish_reason = reason
        except (ValueError, TypeError):
            pass
        return body

    def choose(self, public: dict, rgb: np.ndarray | None, tools: list, timeout_s=90) -> ToolCall:
        if self._pending_user is not None:
            raise ModelProtocolError("Finish the pending tool turn before requesting another decision.")
        self.raw_usage = self.response_finish_reason = self.last_decision = None
        self._pending_decision = None
        image = _png_url(rgb)
        try:
            # Wrapping the public input prevents APIModel's Go2 memory-v1 repair
            # path from interpreting any benchmark field called memory_state.
            decision = APIModel.decide(self, {"input": public}, image, tools, [], timeout_s=timeout_s)
            self.last_decision = self._pending_decision = decision
            if len(decision.calls) != 1:
                raise ModelProtocolError("Choose exactly one available tool.")
            call = decision.calls[0]
            if call.name not in {tool["function"]["name"] for tool in tools}:
                raise ModelProtocolError("The requested tool is not available; no tool executed.")
            if not self.retain_context:
                self._pending_user = self._pending_decision = None
            return call
        except ModelError as exc:
            self._complete({"status": "error", "error_code": exc.code, "executed": False})
            raise

    def _complete(self, feedback):
        if self._pending_user is None:
            return
        if self.retain_context:
            # Serialize the entire block before mutating history, including on
            # rejected multi-call responses, so no tool call can be orphaned.
            block = [deepcopy(self._pending_user)]
            decision = self._pending_decision
            if decision is None:
                block.append({"role": "user", "content": _json_text({
                    "event": "request_failed_without_executed_tool", "feedback": feedback})})
            else:
                calls = [{"id": call.call_id, "type": "function", "function": {
                    "name": call.name, "arguments": _json_text(call.arguments)}} for call in decision.calls]
                assistant = {"role": "assistant", "content": decision.assistant_content, "tool_calls": calls}
                if decision.reasoning_content is not None:
                    assistant["reasoning_content"] = decision.reasoning_content
                block.append(assistant)
                block.extend({"role": "tool", "tool_call_id": call.call_id,
                              "content": _json_text(feedback)} for call in decision.calls)
            _json_text(block)
            self._dialogue.extend(block)
            self._turns += 1
        self._pending_user = self._pending_decision = None

    def finish(self, call, feedback):
        if not self.retain_context:
            return
        decision = self._pending_decision
        if (self._pending_user is None or decision is None or len(decision.calls) != 1
                or decision.calls[0] != call):
            raise ModelProtocolError("Feedback must match the pending tool call.")
        self._complete(feedback)

    def context_stats(self):
        images = sum(sum(part.get("type") == "image_url" for part in message["content"])
                     for message in self._dialogue if isinstance(message.get("content"), list))
        return {"turns": self._turns, "images": images, "messages": len(self._dialogue)}

    def close(self):
        """The base client closes its HTTP connection after each request."""
