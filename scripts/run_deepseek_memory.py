"""Launch the memory-v1 planner with the official DeepSeek thinking adapter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from go2_demo.__main__ import main
from go2_demo.models import APIModel


send_request = APIModel._request


async def send_compatible_request(self, payload, timeout_s):
    if (self.endpoint == "https://api.deepseek.com/chat/completions"
            and payload.get("thinking") == {"type": "enabled"}
            and payload.get("tool_choice") == "required"):
        payload = {**payload, "tool_choice": "auto"}
    return await send_request(self, payload, timeout_s)


if __name__ == "__main__":
    APIModel._request = send_compatible_request
    raise SystemExit(main(["--memory-v1", *sys.argv[1:]]))
