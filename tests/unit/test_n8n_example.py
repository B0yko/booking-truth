"""The importable n8n proxy workflow and its harness agent.yaml stay consistent with each other."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from booking_truth.harness.adapters import load_agent_config

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / "examples" / "n8n" / "booking-agent-proxy.json"
AGENT_CONFIG_PATH = ROOT / "examples" / "agents" / "n8n.yaml"


def workflow() -> dict[str, Any]:
    data = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def node(data: dict[str, Any], node_type: str) -> dict[str, Any]:
    matches = [n for n in data["nodes"] if n["type"] == node_type]
    assert len(matches) == 1, f"expected exactly one {node_type} node, found {len(matches)}"
    return matches[0]


def test_workflow_is_valid_json_with_the_required_top_level_shape() -> None:
    data = workflow()
    assert isinstance(data["name"], str)
    assert data["name"]
    assert isinstance(data["id"], str)
    assert data["id"]
    assert isinstance(data["nodes"], list)
    assert data["nodes"]
    assert isinstance(data["connections"], dict)
    assert data["connections"]
    assert data["active"] is False  # activation happens through `publish:workflow`, not an imported flag


def test_workflow_has_exactly_the_three_expected_node_types() -> None:
    data = workflow()
    types = sorted(n["type"] for n in data["nodes"])
    assert types == [
        "n8n-nodes-base.httpRequest",
        "n8n-nodes-base.respondToWebhook",
        "n8n-nodes-base.webhook",
    ]


def test_webhook_node_is_a_post_with_the_documented_path_and_responds_via_the_third_node() -> None:
    data = workflow()
    webhook = node(data, "n8n-nodes-base.webhook")
    params = webhook["parameters"]
    assert params["httpMethod"] == "POST"
    assert params["path"] == "booking-agent"
    assert params["responseMode"] == "responseNode"


def test_http_request_node_forwards_the_body_with_channel_webhook_and_tolerates_4xx_5xx() -> None:
    data = workflow()
    http = node(data, "n8n-nodes-base.httpRequest")
    params = http["parameters"]
    assert params["method"] == "POST"
    body = params["jsonBody"]
    assert "$json.body" in body
    assert '"webhook"' in body  # channel is overridden to "webhook" before forwarding
    response_options = params["options"]["response"]["response"]
    assert response_options["neverError"] is True
    assert response_options["fullResponse"] is True


def test_respond_to_webhook_returns_the_agents_body_and_status() -> None:
    data = workflow()
    respond = node(data, "n8n-nodes-base.respondToWebhook")
    params = respond["parameters"]
    assert params["respondWith"] == "json"
    assert "$json.body" in params["responseBody"]
    assert "$json.statusCode" in params["options"]["responseCode"]


def test_nodes_are_wired_webhook_then_http_request_then_respond() -> None:
    data = workflow()
    connections = data["connections"]
    assert connections["Webhook"]["main"][0][0]["node"] == "HTTP Request"
    assert connections["HTTP Request"]["main"][0][0]["node"] == "Respond to Webhook"


def test_workflow_carries_no_instance_id_and_no_credentials() -> None:
    data = workflow()
    assert "meta" not in data, "meta.instanceId must be stripped before committing an exported workflow"
    for n in data["nodes"]:
        assert "credentials" not in n, f"{n['name']} must not reference any stored credential"
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "instanceId" not in raw


def test_the_url_and_bearer_are_environment_expressions_not_literal_values() -> None:
    data = workflow()
    http = node(data, "n8n-nodes-base.httpRequest")
    params = http["parameters"]
    assert params["url"] == "={{ $env.BOOKING_AGENT_URL }}"
    headers = {h["name"]: h["value"] for h in params["headerParameters"]["parameters"]}
    assert headers["Authorization"] == "=Bearer {{ $env.BT_API_KEY }}"
    # No literal host, port or secret anywhere in the committed file.
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "localhost" not in raw
    assert "127.0.0.1" not in raw


def test_node_type_versions_are_pinned_to_values_valid_for_the_documented_n8n_version() -> None:
    # Versions confirmed against n8n 2.40.7 node source: see docs/n8n.md.
    data = workflow()
    assert node(data, "n8n-nodes-base.webhook")["typeVersion"] in (1, 1.1, 2, 2.1)
    assert node(data, "n8n-nodes-base.httpRequest")["typeVersion"] in (1, 2, 3, 4, 4.1, 4.2, 4.3, 4.4, 4.5)
    assert node(data, "n8n-nodes-base.respondToWebhook")["typeVersion"] in (1, 1.1, 1.2, 1.3, 1.4, 1.5)


def test_harness_agent_config_targets_the_workflows_webhook_path() -> None:
    webhook_path = node(workflow(), "n8n-nodes-base.webhook")["parameters"]["path"]
    config = load_agent_config(AGENT_CONFIG_PATH)
    assert urlparse(config.url).path == f"/webhook/{webhook_path}"
    # The workflow forwards the incoming body as is, so the agent.yaml body matches the bundled protocol.
    assert config.body["message"] == "{{message}}"
    assert config.body["lead"]["email"] == "{{lead.email}}"
    assert config.response.reply_path == "reply"
    assert config.response.version_path == "agent_version"
    assert config.session_mode == "session_id"
