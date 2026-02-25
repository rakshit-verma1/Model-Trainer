"""
gemini_service.py — Gemini AI integration for architecture suggestions.

All Gemini calls are funnelled through this module.
Responses are validated: only real torch.nn class names are accepted.
If Gemini suggests a module that doesn't exist, it generates custom code.
"""
import json
import os
import re
from pathlib import Path

import torch.nn as nn


def _get_api_key() -> str:
    """Read API key lazily — checks env var first, then .env file."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    from django.conf import settings
    env_path = Path(settings.BASE_DIR) / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    os.environ["GEMINI_API_KEY"] = key
                    return key
    return ""


# All valid torch.nn module names (used for validation)
_VALID_NN_MODULES = {
    name for name, obj in vars(nn).items()
    if isinstance(obj, type) and issubclass(obj, nn.Module) and name[0].isupper()
}

SYSTEM_PROMPT = """You are an expert PyTorch model architect. You help users design neural network architectures.

RULES:
1. You MUST ONLY use valid torch.nn module class names (e.g. Linear, Conv2d, LSTM, TransformerEncoderLayer, etc.)
2. Each layer is a JSON object with "type" (the nn.Module class name) and constructor kwargs.
3. For layers that need special forward() handling (like LSTM, GRU, RNN), add a "_mode" field:
   - "_mode": "rnn" for RNN/LSTM/GRU layers (output is tuple, take first element)
   - "_mode": "select_last" to select the last timestep from sequence output  
   - "_mode": "permute" with "_permute_dims" to rearrange dimensions
   - "_mode": "unsqueeze" with "_dim" to add a dimension
   - "_mode": "squeeze" with "_dim" to remove a dimension
   - "_mode": "reshape" with "_shape" to reshape tensor
4. SKIP/RESIDUAL CONNECTIONS: To add the output from a previous layer to the current layer's output,
   add "_skip_from": <layer_index> (0-based). The shapes must match for element-wise addition.
   Example: layer 5 with "_skip_from": 2 means output_5 = layer_5(x) + output_of_layer_2.
   Use this for ResNet-style skip connections, residual blocks, etc.
5. Always return valid JSON. Never include comments in the JSON.
6. If you need a module that doesn't exist in torch.nn, you can define a CUSTOM module by setting:
   - "type": "Custom" 
   - "_custom_name": "MyModuleName"
   - "_custom_code": "class MyModuleName(nn.Module):\\n    def __init__(self, ...):\\n        ...\\n    def forward(self, x):\\n        ...\\n        return x"
   And include all constructor kwargs as usual. The code MUST be valid Python using only torch and torch.nn.
7. Output the final_output_size (number of output features) that matches the task.

RESPONSE FORMAT — return ONLY this JSON, nothing else:
{
  "layers": [
    {"type": "Conv2d", "in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "padding": 1, "_skip_from": 0},
    {"type": "ReLU"},
    {"type": "Linear", "in_features": 256, "out_features": 10}
  ],
  "task_type": "classification",
  "loss": "CrossEntropyLoss",
  "optimizer": "Adam",
  "lr": 0.001,
  "reasoning": "Brief explanation of the architecture choices"
}
"""


def _get_client():
    """Lazy-init Gemini client."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Set it to your Google AI API key or add it to .env file."
        )
    from google import genai
    return genai.Client(api_key=api_key)


def _extract_json(text: str) -> dict:
    """Extract JSON from Gemini response (may be wrapped in markdown fences)."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text.strip())


def _validate_layers(layers: list[dict]) -> list[str]:
    """Validate that all layer types exist in torch.nn (unless Custom). Returns list of errors."""
    errors = []
    for i, spec in enumerate(layers):
        layer_type = spec.get("type", "")
        if layer_type == "Custom":
            # Custom modules are allowed if they have code
            if not spec.get("_custom_code"):
                errors.append(f"Layer {i+1}: Custom module missing '_custom_code'")
            continue
        if layer_type and layer_type not in _VALID_NN_MODULES:
            errors.append(
                f"Layer {i+1}: '{layer_type}' is not a valid torch.nn module. "
                f"Did you mean one of: {_find_similar(layer_type)}?"
            )
    return errors


def _find_similar(name: str, n: int = 3) -> str:
    """Find similar module names for error messages."""
    name_lower = name.lower()
    scored = sorted(
        _VALID_NN_MODULES,
        key=lambda x: (0 if name_lower in x.lower() else 1, len(x))
    )
    return ", ".join(scored[:n])


def suggest_architecture(description: str, task_type: str = "classification") -> dict:
    """
    Ask Gemini to generate a PyTorch architecture from a text description.
    Returns dict with 'layers', 'task_type', 'loss', 'optimizer', 'lr', 'reasoning'.
    On error, returns dict with 'error' key.
    """
    try:
        client = _get_client()
        prompt = (
            f"Design a PyTorch neural network for this task:\n\n"
            f"Task type: {task_type}\n"
            f"Description: {description}\n\n"
            f"Return the architecture as JSON following the format in your instructions."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={"system_instruction": SYSTEM_PROMPT},
            contents=prompt,
        )
        result = _extract_json(response.text)

        if "error" in result:
            return result
        layers = result.get("layers", [])
        errors = _validate_layers(layers)
        if errors:
            return {"error": "Gemini generated invalid layers:\n" + "\n".join(errors)}

        return result

    except json.JSONDecodeError:
        return {"error": "Gemini returned invalid JSON. Please try again."}
    except Exception as e:
        return {"error": str(e)}


def parse_pdf_and_suggest(pdf_bytes: bytes) -> dict:
    """
    Extract text from a PDF and ask Gemini to design an architecture based on it.
    """
    try:
        import PyPDF2
        import io

        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)

        if not text_parts:
            return {"error": "Could not extract any text from the PDF."}

        pdf_text = "\n".join(text_parts)
        if len(pdf_text) > 8000:
            pdf_text = pdf_text[:8000] + "\n...[truncated]"

        client = _get_client()
        prompt = (
            f"Based on the following document/paper, design a PyTorch neural network "
            f"architecture that implements or is inspired by the described model.\n\n"
            f"DOCUMENT CONTENT:\n{pdf_text}\n\n"
            f"Return the architecture as JSON following the format in your instructions. "
            f"If the paper describes a model that cannot be built with standard torch.nn "
            f"modules alone, use Custom modules with _custom_code to implement them."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={"system_instruction": SYSTEM_PROMPT},
            contents=prompt,
        )
        result = _extract_json(response.text)

        if "error" in result:
            return result
        layers = result.get("layers", [])
        errors = _validate_layers(layers)
        if errors:
            return {"error": "Gemini generated invalid layers:\n" + "\n".join(errors)}

        return result

    except json.JSONDecodeError:
        return {"error": "Gemini returned invalid JSON. Please try again."}
    except Exception as e:
        return {"error": str(e)}


def help_with_layer(layer_type: str, context: str = "") -> dict:
    """
    Ask Gemini to explain a layer and suggest parameters.
    Returns dict with 'help' text and optionally 'suggested_params'.
    """
    try:
        client = _get_client()
        prompt = (
            f"Explain the PyTorch nn.{layer_type} module concisely.\n"
            f"- What it does\n- Key parameters and recommended values\n"
            f"- When to use it\n"
        )
        if context:
            prompt += f"\nContext from the user: {context}\n"
        prompt += (
            "\nReturn JSON: {\"help\": \"explanation text\", "
            "\"suggested_params\": {\"param\": value, ...}}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={"system_instruction": SYSTEM_PROMPT},
            contents=prompt,
        )
        return _extract_json(response.text)

    except Exception as e:
        return {"error": str(e)}
