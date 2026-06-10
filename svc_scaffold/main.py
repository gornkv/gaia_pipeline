from __future__ import annotations

import os

from svc_scaffold.api.openai_compatible import create_app as create_openai_compatible_app
from svc_scaffold.clients.openai_compatible import OpenAICompatibleModelClient
from svc_scaffold.core import Scaffold


model_client = OpenAICompatibleModelClient(
    os.environ["BASE_MODEL_API_BASE_URL"],
    os.environ["BASE_MODEL_API_KEY"],
    os.environ["BASE_MODEL_NAME"],
)
scaffold = Scaffold(model_client, max(1, int(os.getenv("SCAFFOLD_BON_CANDIDATES", "3"))))

app = create_openai_compatible_app(scaffold, model_client)
