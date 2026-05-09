"""
Tests that conversation_id is propagated into context loading and streaming setup.

Regression coverage for named conversations accidentally using the default message path.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from letta.agents.letta_agent_v2 import LettaAgentV2
from letta.data_sources.redis_client import NoopAsyncRedisClient
from letta.schemas.conversation import CreateConversation
from letta.schemas.enums import MessageRole
from letta.schemas.letta_request import LettaStreamingRequest
from letta.schemas.message import MessageCreate
from letta.services.conversation_manager import ConversationManager
from letta.services.streaming_service import StreamingService


def _mock_step_noop(self, *args, **kwargs):
    """Return an async generator that ends the step loop without yielding."""

    async def _gen():
        self.should_continue = False
        if False:  # pragma: no cover
            yield None

    return _gen()


@pytest.mark.asyncio
async def test_letta_agent_v2_step_passes_conversation_id_to_prepare(server, default_user, sarah_agent):
    agent = await server.agent_manager.get_agent_by_id_async(
        sarah_agent.id,
        default_user,
        include_relationships=["memory", "multi_agent_group", "sources", "tool_exec_environment_variables", "tools", "tags"],
    )
    loop = LettaAgentV2(agent_state=agent, actor=default_user)
    conv_id = "conv-test-propagation"

    with patch(
        "letta.agents.letta_agent_v2._prepare_in_context_messages_no_persist_async",
        new_callable=AsyncMock,
    ) as mock_prepare:
        mock_prepare.return_value = ([], [])
        with patch.object(LettaAgentV2, "_step", new=_mock_step_noop):
            with patch.object(LettaAgentV2, "summarize_conversation_history", new_callable=AsyncMock, return_value=[]):
                await loop.step(
                    [MessageCreate(role=MessageRole.user, content="hi")],
                    max_steps=1,
                    conversation_id=conv_id,
                )

    mock_prepare.assert_awaited_once()
    _, kwargs = mock_prepare.call_args
    assert kwargs.get("conversation_id") == conv_id


@pytest.mark.asyncio
async def test_letta_agent_v2_stream_passes_conversation_id_to_prepare(server, default_user, sarah_agent):
    agent = await server.agent_manager.get_agent_by_id_async(
        sarah_agent.id,
        default_user,
        include_relationships=["memory", "multi_agent_group", "sources", "tool_exec_environment_variables", "tools", "tags"],
    )
    loop = LettaAgentV2(agent_state=agent, actor=default_user)
    conv_id = "conv-stream-propagation"

    with patch(
        "letta.agents.letta_agent_v2._prepare_in_context_messages_no_persist_async",
        new_callable=AsyncMock,
    ) as mock_prepare:
        mock_prepare.return_value = ([], [])
        with patch.object(LettaAgentV2, "_step", new=_mock_step_noop):
            with patch.object(LettaAgentV2, "summarize_conversation_history", new_callable=AsyncMock, return_value=[]):
                agen = loop.stream(
                    [MessageCreate(role=MessageRole.user, content="hi")],
                    max_steps=1,
                    conversation_id=conv_id,
                )
                async for _ in agen:
                    pass

    mock_prepare.assert_awaited_once()
    _, kwargs = mock_prepare.call_args
    assert kwargs.get("conversation_id") == conv_id


@pytest.mark.asyncio
async def test_letta_agent_v2_build_request_passes_conversation_id_to_prepare(server, default_user, sarah_agent):
    agent = await server.agent_manager.get_agent_by_id_async(
        sarah_agent.id,
        default_user,
        include_relationships=["memory", "multi_agent_group", "sources", "tool_exec_environment_variables", "tools", "tags"],
    )
    loop = LettaAgentV2(agent_state=agent, actor=default_user)
    conv_id = "conv-preview-propagation"

    with patch(
        "letta.agents.letta_agent_v2._prepare_in_context_messages_no_persist_async",
        new_callable=AsyncMock,
    ) as mock_prepare:
        mock_prepare.return_value = ([], [])
        with patch.object(LettaAgentV2, "_step", new=_mock_step_noop):
            await loop.build_request(
                [MessageCreate(role=MessageRole.user, content="hi")],
                conversation_id=conv_id,
            )

    mock_prepare.assert_awaited_once()
    _, kwargs = mock_prepare.call_args
    assert kwargs.get("conversation_id") == conv_id


@pytest.mark.asyncio
async def test_streaming_service_uses_v3_with_conversation_id(server, default_user, sarah_agent):
    """When conversation_id is set, create_agent_stream must construct LettaAgentV3 with that id."""
    agent = await server.agent_manager.get_agent_by_id_async(
        sarah_agent.id,
        default_user,
        include_relationships=["memory", "multi_agent_group", "sources", "tool_exec_environment_variables", "tools", "tags"],
    )
    cm = ConversationManager()
    conversation = await cm.create_conversation(
        agent_id=agent.id,
        conversation_create=CreateConversation(),
        actor=default_user,
    )

    sentinel_loop = object()

    async def fake_error_aware_stream(self, agent_loop, *args, **kwargs):
        assert agent_loop is sentinel_loop
        yield "data: [DONE]\n\n"

    mock_v3_cls = MagicMock(return_value=sentinel_loop)

    svc = StreamingService(server)
    with patch("letta.agents.letta_agent_v3.LettaAgentV3", mock_v3_cls):
        with patch.object(StreamingService, "_create_error_aware_stream", new=fake_error_aware_stream):
            with patch("letta.services.streaming_service.get_redis_client", new_callable=AsyncMock, return_value=NoopAsyncRedisClient()):
                with patch("letta.services.streaming_service.settings") as mock_settings:
                    mock_settings.track_agent_run = False
                    mock_settings.enable_cancellation_aware_streaming = False
                    mock_settings.enable_keepalive = False
                    mock_settings.keepalive_interval = 30

                    request = LettaStreamingRequest(
                        messages=[MessageCreate(role=MessageRole.user, content="hello")],
                        streaming=True,
                        stream_tokens=False,
                        include_pings=False,
                        background=False,
                        max_steps=1,
                    )

                    await svc.create_agent_stream(
                        agent_id=agent.id,
                        actor=default_user,
                        request=request,
                        run_type="send_conversation_message",
                        conversation_id=conversation.id,
                    )

    mock_v3_cls.assert_called_once()
    call_kw = mock_v3_cls.call_args.kwargs
    assert call_kw["conversation_id"] == conversation.id
    assert call_kw["agent_state"].id == agent.id
    assert call_kw["actor"] is default_user
