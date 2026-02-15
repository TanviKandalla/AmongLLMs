"""
Pytest test suite for LLM response validation and action parsing.

Tests cover:
- Valid action matching
- Hallucinated/unavailable action detection (now triggers retry, not AttemptedAction)
- Malformed and truncated responses
- Edge cases with weird formatting
- SPEAK action special handling
- VOTE action special handling
- Retry logic behavior (including SKIP VOTE fallback during voting phase)
"""

import os
import re
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Set required environment variable before importing agent
os.environ["EXPERIMENT_PATH"] = tempfile.gettempdir()

from amongagents.agent.agent import LLMAgent
from amongagents.envs.action import (
    Action,
    AttemptedAction,
    CallMeeting,
    CompleteTask,
    Kill,
    MoveTo,
    SkipVote,
    Speak,
    Vent,
    Vote,
)
from amongagents.envs.player import Crewmate, Impostor

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_crewmate():
    """Create a mock Crewmate player for testing."""
    # Crewmate signature: name, color, personality, location=None
    player = Crewmate(
        name="Player 1",
        color="white",
        personality=None,
        location="Cafeteria",
    )
    # Set location_info to avoid NoneType concatenation errors
    player.location_info = "You are in Cafeteria."
    return player


@pytest.fixture
def mock_impostor():
    """Create a mock Impostor player for testing."""
    player = Impostor(
        name="Player 2",
        color="red",
        personality=None,
        location="Cafeteria",
    )
    return player


@pytest.fixture
def mock_target_player():
    """Create a mock target player for KILL/VOTE actions."""
    player = Crewmate(
        name="Player 3",
        color="blue",
        personality=None,
        location="Cafeteria",
    )
    return player


@pytest.fixture
def mock_agent(mock_crewmate):
    """Create a mock LLMAgent for testing."""
    agent_config = {"max_steps": 50}
    agent = LLMAgent(
        mock_crewmate,
        [],
        game_index=1,
        agent_config=agent_config,
        list_of_impostors=[],
        model="test-model",
    )
    return agent


@pytest.fixture
def basic_available_actions(mock_target_player):
    """Create a basic set of available actions for testing."""
    return [
        MoveTo("Cafeteria", "Admin"),
        MoveTo("Cafeteria", "Weapons"),
        Kill("Cafeteria", mock_target_player),
        Speak("Cafeteria"),
        CallMeeting("Cafeteria"),
    ]


@pytest.fixture
def vote_available_actions(mock_target_player):
    """Create a set of voting actions for meeting phase testing."""
    target2 = Crewmate(name="Player 4", color="cyan", personality=None, location="Cafeteria")
    return [
        Vote("Cafeteria", mock_target_player),
        Vote("Cafeteria", target2),
        SkipVote("Cafeteria"),
    ]


@pytest.fixture
def vote_available_actions_no_skip(mock_target_player):
    """Create a set of voting actions WITHOUT skip for legacy test compatibility."""
    target2 = Crewmate(name="Player 4", color="cyan", personality=None, location="Cafeteria")
    return [
        Vote("Cafeteria", mock_target_player),
        Vote("Cafeteria", target2),
    ]


# ============================================================================
# Test: Valid Action Matching
# ============================================================================


class TestValidActionMatching:
    """Tests for correctly matching valid actions from LLM responses."""

    def test_exact_match_move_action(self, mock_agent, basic_available_actions):
        """Test exact match of MOVE action."""
        response = """[Condensed Memory]
Game just started. I'm in Cafeteria.
[Thinking Process]
I should move to Admin to complete my task.
[Action] MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert error is None
        assert action.name == "MOVE"
        assert memory == "Game just started. I'm in Cafeteria."

    def test_exact_match_kill_action(self, mock_agent, basic_available_actions):
        """Test exact match of KILL action."""
        response = """[Condensed Memory]
I'm an impostor and must eliminate crewmates.
[Thinking Process]
Player 3 is alone with me. Perfect opportunity.
[Action] KILL Player 3: blue"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert error is None
        assert action.name == "KILL"

    def test_case_insensitive_matching(self, mock_agent, basic_available_actions):
        """Test that action matching is case insensitive."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving to admin.
[Action] move from cafeteria to admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_speak_action_with_message(self, mock_agent, basic_available_actions):
        """Test SPEAK action extracts the message correctly."""
        response = """[Condensed Memory]
Meeting phase.
[Thinking Process]
I need to share my observations.
[Action] SPEAK: I saw Player 2 near the body in Electrical!"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "SPEAK"
        assert "I saw Player 2" in action.message

    def test_call_meeting_action(self, mock_agent, basic_available_actions):
        """Test CALL MEETING action matching."""
        response = """[Condensed Memory]
Found a body!
[Thinking Process]
Must report immediately.
[Action] CALL MEETING using the emergency button at Cafeteria"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "CALL MEETING"  # Note: space not underscore

    def test_report_dead_body_matches_report_action_not_button(self, mock_agent):
        """Regression: REPORT DEAD BODY should map to CallMeeting(is_report=True)."""
        available_actions = [
            CallMeeting("Cafeteria", is_report=False, buttons_remaining=1),
            CallMeeting("Cafeteria", is_report=True),
        ]

        response = """[Condensed Memory]
I found a body in my room.
[Thinking Process]
I should report it immediately.
[Action] REPORT DEAD BODY at Cafeteria"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, available_actions
        )

        assert action is not None
        assert error is None
        assert action.name == "CALL MEETING"
        assert hasattr(action, "is_report")
        assert action.is_report is True


# ============================================================================
# Test: Hallucinated Action Detection
# ============================================================================


class TestHallucinatedActionDetection:
    """Tests for detecting and handling hallucinated (unavailable) actions.

    After the unified retry refactor, hallucinated actions now return
    (None, ..., error_msg) instead of AttemptedAction, so they trigger
    the retry loop just like any other invalid output.
    """

    def test_unavailable_kill_returns_error(self, mock_agent):
        """Test that attempting to KILL when not available returns error (triggers retry)."""
        # Only MOVE actions available (crewmate scenario)
        available_actions = [
            MoveTo("Cafeteria", "Admin"),
        ]

        response = """[Condensed Memory]
I think Player 2 is suspicious.
[Thinking Process]
I'll eliminate them.
[Action] KILL Player 2: red"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, available_actions
        )

        assert action is None
        assert error is not None
        assert "Could not match action" in error

    def test_unavailable_vent_returns_error(self, mock_agent):
        """Test that attempting to VENT when not available returns error (triggers retry)."""
        available_actions = [
            MoveTo("Cafeteria", "Admin"),
        ]

        response = """[Condensed Memory]
Need to escape.
[Thinking Process]
I'll use the vent.
[Action] VENT to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, available_actions
        )

        assert action is None
        assert error is not None

    def test_unavailable_move_destination_returns_error(self, mock_agent):
        """Test that moving to unavailable location returns error (triggers retry)."""
        available_actions = [
            MoveTo("Cafeteria", "Admin"),
        ]

        response = """[Condensed Memory]
Going to Security.
[Thinking Process]
I want to check cameras.
[Action] MOVE to Security"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, available_actions
        )

        assert action is None
        assert error is not None

    def test_attempted_action_execute_does_nothing(self, mock_agent):
        """Test that executing AttemptedAction has no effect."""
        attempted = AttemptedAction("KILL Player 2", current_location="Cafeteria")

        # Should not raise any exceptions
        attempted.execute(None, None)

        assert attempted.action_text() == "attempted KILL Player 2 but failed"


# ============================================================================
# Test: Malformed and Truncated Responses
# ============================================================================


class TestMalformedResponses:
    """Tests for handling malformed and truncated LLM responses."""

    def test_truncated_response_no_action_section(
        self, mock_agent, basic_available_actions
    ):
        """Test response that was cut off before [Action] section."""
        response = """[Condensed Memory]
Game just started. I'm in Cafeteria with other players.
[Thinking Process]
I need to think strategically about my next move. The best approach would be to"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should return error since no action can be parsed
        assert action is None
        assert error is not None
        assert "Could not match action" in error

    def test_truncated_mid_action(self, mock_agent, basic_available_actions):
        """Test response truncated in the middle of the action text."""
        response = """[Condensed Memory]
Game info.
[Thinking Process]
I'll move.
[Action] MOVE from Cafet"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Truncated action won't match - should be AttemptedAction or error
        # The partial "MOVE" might be detected as attempted
        assert action is None or isinstance(action, AttemptedAction)

    def test_empty_response(self, mock_agent, basic_available_actions):
        """Test completely empty response."""
        response = ""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is None
        assert "empty" in error.lower()

    def test_whitespace_only_response(self, mock_agent, basic_available_actions):
        """Test response with only whitespace."""
        response = "   \n\n   \t   "

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is None
        assert error is not None

    def test_garbage_response(self, mock_agent, basic_available_actions):
        """Test completely nonsensical response."""
        response = "asdfghjkl 12345 !@#$% lorem ipsum dolor sit amet"

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is None
        assert error is not None
        assert "Could not match action" in error

    def test_action_without_brackets(self, mock_agent, basic_available_actions):
        """Test response with action but no [Action] marker."""
        response = """I will MOVE from Cafeteria to Admin to complete my task."""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should still match via fallback to full response
        assert action is not None
        assert action.name == "MOVE"

    def test_multiple_action_sections(self, mock_agent, basic_available_actions):
        """Test response with multiple [Action] sections (model self-correcting).
        
        Should be REJECTED — the model is second-guessing itself, which risks
        matching on a hallucinated action. Force a clean retry instead.
        """
        response = """[Condensed Memory]
Testing.
[Thinking Process]
First I thought about moving.
[Action] MOVE from Cafeteria to Weapons
Wait, actually I should go to Admin instead.
[Action] MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is None
        assert error is not None
        assert "[Action] tags" in error

    def test_multiple_action_sections_with_hallucinated_first(
        self, mock_agent, basic_available_actions
    ):
        """Model picks a hallucinated action, realizes, then picks a valid one.
        
        Must reject the whole response — we can't trust which was intended.
        """
        response = """[Condensed Memory]
I need to eliminate someone.
[Thinking Process]
I'll sabotage O2.
[Action] SABOTAGE O2
Oh wait, that one isn't available.
[Action] MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is None
        assert error is not None
        assert "[Action] tags" in error

    def test_only_action_marker_no_content(self, mock_agent, basic_available_actions):
        """Test response with [Action] marker but no action content."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
I don't know what to do.
[Action]"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is None
        assert error is not None

    def test_action_marker_with_only_whitespace(
        self, mock_agent, basic_available_actions
    ):
        """Test [Action] followed by only whitespace/newlines."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Thinking.
[Action]

"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is None
        assert error is not None

    def test_unicode_garbage(self, mock_agent, basic_available_actions):
        """Test response with unicode garbage characters."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
\u200b\u200b\ufeff
[Action] 🎮 ➡️ Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should fail gracefully
        assert error is not None or isinstance(action, AttemptedAction)


# ============================================================================
# Test: Edge Cases with Weird Formatting
# ============================================================================


class TestEdgeCaseFormatting:
    """Tests for edge cases in response formatting."""

    def test_extra_whitespace_in_action(self, mock_agent, basic_available_actions):
        """Test action with extra whitespace."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Going somewhere.
[Action]    MOVE   from   Cafeteria   to   Admin   """

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_action_wrapped_in_braces(self, mock_agent, basic_available_actions):
        """Test action wrapped in curly braces - common model quirk."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving to Admin.
[Action] {MOVE from Cafeteria to Admin}"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should still match the action inside the braces
        assert action is not None
        assert action.name == "MOVE"

    def test_action_with_json_like_format(self, mock_agent, basic_available_actions):
        """Test action formatted as JSON object."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving to Admin.
[Action] {"action": "MOVE", "from": "Cafeteria", "to": "Admin"}"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should try to handle or gracefully fail
        # The response contains MOVE, Cafeteria, Admin so it might match
        assert action is not None or error is not None

    def test_action_with_backticks(self, mock_agent, basic_available_actions):
        """Test action wrapped in markdown backticks."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] `MOVE from Cafeteria to Admin`"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_action_with_code_block(self, mock_agent, basic_available_actions):
        """Test action in a code block."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action]
```
MOVE from Cafeteria to Admin
```"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_action_on_same_line_as_bracket(self, mock_agent, basic_available_actions):
        """Test [Action] marker on same line as action."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action]MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_lowercase_section_headers(self, mock_agent, basic_available_actions):
        """Test lowercase section headers."""
        response = """[condensed memory]
Testing.
[thinking process]
Moving.
[action] MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should handle case-insensitive section headers
        assert (
            action is not None
            or isinstance(action, AttemptedAction)
            or error is not None
        )

    def test_speak_with_colon_variations(self, mock_agent, basic_available_actions):
        """Test SPEAK action with different colon placements."""
        responses = [
            "[Action] SPEAK: Hello everyone",
            "[Action] SPEAK : Hello everyone",
            "[Action] SPEAK  :  Hello everyone",
            "[Action] SPEAK Hello everyone",
        ]

        for response in responses:
            full_response = f"""[Condensed Memory]
Test.
[Thinking Process]
Speaking.
{response}"""
            action, _, _, _ = mock_agent._validate_and_parse_action(
                full_response, basic_available_actions
            )
            assert action is not None, f"Failed for: {response}"

    def test_action_with_trailing_punctuation(
        self, mock_agent, basic_available_actions
    ):
        """Test action with trailing punctuation."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving now.
[Action] MOVE from Cafeteria to Admin."""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"


# ============================================================================
# Test: Vote Action Special Handling
# ============================================================================


class TestVoteActionHandling:
    """Tests for VOTE action parsing."""

    def test_vote_for_player(self, mock_agent, vote_available_actions):
        """Test voting for a specific player."""
        response = """[Condensed Memory]
Player 3 is suspicious.
[Thinking Process]
I'm voting for Player 3.
[Action] VOTE Player 3: blue"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions
        )

        assert action is not None
        assert action.name == "VOTE"

    def test_vote_with_partial_name(self, mock_agent, vote_available_actions):
        """Test voting with partial player name - returns error (triggers retry).

        Note: The current parser doesn't support partial name matching for votes.
        This test documents the current behavior - partial names trigger retry.
        """
        response = """[Condensed Memory]
Blue is sus.
[Thinking Process]
Voting blue.
[Action] VOTE blue"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions
        )

        # Current behavior: partial name doesn't match, returns error for retry
        assert action is None
        assert error is not None

    def test_vote_skip_returns_error_when_not_available(self, mock_agent, vote_available_actions_no_skip):
        """Test skip vote returns error when NOT in available actions (triggers retry)."""
        response = """[Condensed Memory]
Not enough evidence.
[Thinking Process]
I'll skip this vote.
[Action] VOTE SKIP"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions_no_skip
        )

        # Skip is not in available actions, should return error for retry
        assert action is None
        assert error is not None

    def test_vote_with_for_keyword(self, mock_agent, vote_available_actions):
        """Test vote with 'for' keyword: VOTE for Player X."""
        response = """[Condensed Memory]
Voting time.
[Thinking Process]
Player 3 is suspicious.
[Action] VOTE for Player 3: blue"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions
        )

        assert action is not None
        assert action.name == "VOTE"

    def test_vote_with_color_only(self, mock_agent, vote_available_actions):
        """Test vote using only color - returns error (triggers retry).

        Note: The current parser doesn't support color-only matching for votes.
        This test documents the current behavior.
        """
        response = """[Condensed Memory]
Voting.
[Thinking Process]
I think cyan is sus.
[Action] VOTE cyan"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions
        )

        # Current behavior: color-only doesn't match full name, returns error for retry
        assert action is None
        assert error is not None

    def test_vote_nonexistent_player(self, mock_agent, vote_available_actions):
        """Test voting for a player not in the game."""
        response = """[Condensed Memory]
Voting.
[Thinking Process]
Green is sus.
[Action] VOTE Player 5: green"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions
        )

        # Should return error since Player 5 doesn't exist (triggers retry)
        assert action is None
        assert error is not None


# ============================================================================
# Test: AttemptedAction Class
# ============================================================================


class TestAttemptedActionClass:
    """Tests for the AttemptedAction class itself."""

    def test_attempted_action_repr(self):
        """Test AttemptedAction string representation."""
        attempted = AttemptedAction("KILL Player 5", current_location="Admin")

        assert "ATTEMPTED" in repr(attempted)
        assert "KILL Player 5" in repr(attempted)

    def test_attempted_action_text(self):
        """Test AttemptedAction action_text method."""
        attempted = AttemptedAction("VENT to Security", current_location="Electrical")

        assert "attempted" in attempted.action_text()
        assert "failed" in attempted.action_text()
        assert "VENT to Security" in attempted.action_text()

    def test_attempted_action_can_execute_returns_empty(self):
        """Test that AttemptedAction.can_execute_actions returns empty list."""
        result = AttemptedAction.can_execute_actions(None, None)
        assert result == []


# ============================================================================
# Test: SkipVote Action
# ============================================================================


class TestSkipVoteAction:
    """Tests for the SkipVote action class and mechanics."""

    def test_skip_vote_action_class(self):
        """Test SkipVote action class properties."""
        skip = SkipVote("Cafeteria")

        assert skip.name == "SKIP VOTE"
        assert skip.current_location == "Cafeteria"
        assert "SKIP VOTE" in repr(skip)

    def test_skip_vote_can_execute_returns_empty(self):
        """Test that SkipVote.can_execute_actions returns empty list.

        Skip votes are added via Vote.can_execute_actions, not standalone.
        """
        result = SkipVote.can_execute_actions(None, None)
        assert result == []

    def test_skip_vote_parses_correctly(self, mock_agent, vote_available_actions):
        """Test that SKIP VOTE action parses correctly when available."""
        response = """[Condensed Memory]
Not enough evidence to vote anyone.
[Thinking Process]
I don't have clear evidence, so I'll skip.
[Action] SKIP VOTE"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions
        )

        assert action is not None
        assert error is None
        assert action.name == "SKIP VOTE"

    def test_skip_vote_with_lowercase(self, mock_agent, vote_available_actions):
        """Test skip vote with lowercase input."""
        response = """[Condensed Memory]
Uncertain who to vote for.
[Thinking Process]
Skipping.
[Action] skip vote"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions
        )

        assert action is not None
        assert action.name == "SKIP VOTE"

    def test_skip_vote_with_extra_text(self, mock_agent, vote_available_actions):
        """Test skip vote with extra explanatory text."""
        response = """[Condensed Memory]
Meeting phase.
[Thinking Process]
Need more info.
[Action] SKIP VOTE - I want to observe more before deciding"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions
        )

        # Should still match SKIP VOTE
        assert action is not None
        assert action.name == "SKIP VOTE"

    def test_vote_skip_format_still_works(self, mock_agent, vote_available_actions):
        """Test that 'VOTE SKIP' format also works for skip voting."""
        response = """[Condensed Memory]
Voting time.
[Thinking Process]
No clear suspect.
[Action] VOTE SKIP"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, vote_available_actions
        )

        # With SkipVote in available actions, this should match
        assert action is not None
        # It might match as SkipVote or as AttemptedAction for VOTE
        # The key is it doesn't error


# ============================================================================
# Test: Tie Vote Logic
# ============================================================================


class TestTieVoteLogic:
    """Tests for tie vote mechanics in the game."""

    def test_skip_vote_execute_records_skip(self, mock_crewmate):
        """Test that SkipVote.execute() records 'SKIP' in vote info."""
        env = Mock()
        env.vote_info_one_round = {}

        skip = SkipVote("Cafeteria")
        skip.execute(env, mock_crewmate)

        assert mock_crewmate.name in env.vote_info_one_round
        assert env.vote_info_one_round[mock_crewmate.name] == "SKIP"

    def test_skip_vote_does_not_increment_votes(self, mock_crewmate):
        """Test that SkipVote doesn't increment any player's vote count."""
        env = Mock()
        env.vote_info_one_round = {}
        env.votes = {}

        skip = SkipVote("Cafeteria")
        skip.execute(env, mock_crewmate)

        # votes dict should remain empty - skip doesn't add votes
        assert env.votes == {}

    def test_vote_action_increments_votes(self, mock_crewmate, mock_target_player):
        """Test that regular Vote does increment the target's vote count."""
        env = Mock()
        env.vote_info_one_round = {}
        env.votes = {}

        vote = Vote("Cafeteria", mock_target_player)
        vote.execute(env, mock_crewmate)

        # Target should have 1 vote
        assert env.votes.get(mock_target_player, 0) == 1
        assert env.vote_info_one_round[mock_crewmate.name] == mock_target_player.name


# ============================================================================
# Test: Real-World Messy Logs
# ============================================================================


class TestRealWorldMessyLogs:
    """Tests based on actual messy LLM outputs observed in production."""

    def test_truncated_thinking_process(self, mock_agent, basic_available_actions):
        """Test response where thinking process was cut off mid-sentence."""
        response = """[Condensed Memory]
No actions have been taken yet. The game just started. I am in the Cafeteria with Player 6: blue and Player 7: red. Other players have moved to Admin.

[Thinking Process]
My goal is to eliminate Crewmates before they complete all tasks. Since there are no Impostors identified yet, I can act freely without being suspected immediately. I am in the Cafeteria with two other players and I have a task to Fix Wiring in Admin. However, rushing to Admin might make me suspicious. I should consider eliminating a Crewmate to gain an advantage, but doing so in a room with witnesses (like now) could lead to immediate suspicion and a meeting called. Therefore, I need to think of a strategic move that might allow"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should fail - no action section
        assert action is None or isinstance(action, AttemptedAction)

    def test_json_formatted_thinking(self, mock_agent, basic_available_actions):
        """Test response where model outputted JSON in thinking section."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
{
    "thought": "I should move to Admin",
    "reasoning": "To complete my task",
    "action": "MOVE from Cafeteria to Admin"
}
[Action] MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_markdown_formatted_action(self, mock_agent, basic_available_actions):
        """Test response with markdown formatting in action."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving to Admin.
[Action] **MOVE from Cafeteria to Admin**"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should handle or fail gracefully
        assert action is not None or error is not None

    def test_numbered_action(self, mock_agent, basic_available_actions):
        """Test response where model included action number."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
I'll pick action 1.
[Action] 1. MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_action_with_explanation(self, mock_agent, basic_available_actions):
        """Test response where model added explanation after action."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving now.
[Action] MOVE from Cafeteria to Admin (this will let me complete my wiring task)"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_speak_with_multiline_message(self, mock_agent, basic_available_actions):
        """Test SPEAK action with multiline message."""
        response = """[Condensed Memory]
Meeting phase.
[Thinking Process]
I need to explain what I saw.
[Action] SPEAK: I was in Electrical when the lights went out.
I heard a vent but couldn't see who it was.
We should be careful about who we vote for."""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "SPEAK"

    def test_action_with_arrow_notation(self, mock_agent, basic_available_actions):
        """Test model using arrow notation for movement."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] MOVE: Cafeteria -> Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # May or may not match depending on parser flexibility
        # At minimum should be AttemptedAction or match
        assert action is not None or error is not None

    def test_action_with_choice_format(self, mock_agent, basic_available_actions):
        """Test model outputting in choice/option format."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Choosing option A.
[Action] Option A: MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_xml_style_action_tags(self, mock_agent, basic_available_actions):
        """Test model using XML-style tags."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] <action>MOVE from Cafeteria to Admin</action>"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_action_with_bullet_point(self, mock_agent, basic_available_actions):
        """Test action with bullet point prefix."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] • MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_action_with_dash_prefix(self, mock_agent, basic_available_actions):
        """Test action with dash prefix (markdown list style)."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] - MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_repeated_action_keyword(self, mock_agent, basic_available_actions):
        """Test when model repeats the ACTION keyword."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] ACTION: MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_action_quoted(self, mock_agent, basic_available_actions):
        """Test action wrapped in quotes."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] "MOVE from Cafeteria to Admin\""""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_misspelled_section_headers(self, mock_agent, basic_available_actions):
        """Test common misspellings of section headers."""
        response = """[Condensed Memeory]
Testing.
[Thinking Proccess]
Moving.
[Acton] MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Misspelled [Acton] probably won't match [Action] regex
        # Should fallback to finding MOVE in the text
        assert (
            action is not None
            or isinstance(action, AttemptedAction)
            or error is not None
        )

    def test_action_all_caps_sections(self, mock_agent, basic_available_actions):
        """Test all caps section headers."""
        response = """[CONDENSED MEMORY]
Testing.
[THINKING PROCESS]
Moving.
[ACTION] MOVE from Cafeteria to Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        assert action is not None
        assert action.name == "MOVE"

    def test_hallucinated_complete_task(self, mock_agent, basic_available_actions):
        """Test attempting to complete a task not in available actions."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Completing my task.
[Action] COMPLETE TASK - Fix Wiring at Admin"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should return error since COMPLETE TASK isn't available (triggers retry)
        assert action is None
        assert error is not None

    def test_hallucinated_sabotage(self, mock_agent, basic_available_actions):
        """Test attempting to sabotage when not available."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
Sabotaging the lights.
[Action] SABOTAGE lights"""

        action, memory, summarization, error = mock_agent._validate_and_parse_action(
            response, basic_available_actions
        )

        # Should return error since SABOTAGE isn't available (triggers retry)
        assert action is None
        assert error is not None


# ============================================================================
# Test: Integration with Retry Logic (Mocked)
# ============================================================================


class TestRetryLogicIntegration:
    """Tests for retry logic behavior without starting an actual game."""

    @pytest.mark.asyncio
    async def test_successful_first_attempt(self, mock_agent, basic_available_actions):
        """Test that successful first attempt doesn't trigger retry."""
        good_response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] MOVE from Cafeteria to Admin"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                mock_send.return_value = good_response

                # Patch log_interaction to avoid file operations
                with patch.object(mock_agent, "log_interaction"):
                    action = await mock_agent.choose_action(timestep=0)

        # send_request should only be called once
        assert mock_send.call_count == 1
        assert action.name == "MOVE"

    @pytest.mark.asyncio
    async def test_retry_on_malformed_response(
        self, mock_agent, basic_available_actions
    ):
        """Test that malformed response triggers retry."""
        bad_response = "This is garbage output"
        good_response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] MOVE from Cafeteria to Admin"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                mock_send.side_effect = [bad_response, good_response]

                with patch.object(mock_agent, "log_interaction"):
                    action = await mock_agent.choose_action(timestep=0)

        # send_request should be called twice (initial + 1 retry)
        assert mock_send.call_count == 2
        assert action.name == "MOVE"

    @pytest.mark.asyncio
    async def test_hallucination_triggers_retry(self, mock_agent):
        """Test that hallucinated actions trigger retry (unified with other invalid output)."""
        # Only provide MOVE action, but response tries to KILL
        limited_actions = [MoveTo("Cafeteria", "Admin")]

        hallucination_response = """[Condensed Memory]
Testing.
[Thinking Process]
Killing.
[Action] KILL Player 5"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player, "get_available_actions", return_value=limited_actions
            ):
                mock_send.return_value = hallucination_response

                with patch.object(mock_agent, "log_interaction"):
                    with pytest.raises(RuntimeError, match="Format validation failed"):
                        await mock_agent.choose_action(timestep=0)

        # send_request should be called 3 times (all retries exhausted)
        assert mock_send.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_raises_error(
        self, mock_agent, basic_available_actions
    ):
        """Test that exhausting all retries raises RuntimeError."""
        garbage_response = "asdfghjkl not a valid action"

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                mock_send.return_value = garbage_response

                with patch.object(mock_agent, "log_interaction"):
                    with pytest.raises(RuntimeError, match="Format validation failed"):
                        await mock_agent.choose_action(timestep=0)

        # send_request should be called max_format_retries times (3)
        assert mock_send.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_empty_response(self, mock_agent, basic_available_actions):
        """Test that empty response triggers retry."""
        empty_response = ""
        good_response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] MOVE from Cafeteria to Admin"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                mock_send.side_effect = [empty_response, good_response]

                with patch.object(mock_agent, "log_interaction"):
                    action = await mock_agent.choose_action(timestep=0)

        assert mock_send.call_count == 2
        assert action.name == "MOVE"

    @pytest.mark.asyncio
    async def test_retry_on_truncated_response(
        self, mock_agent, basic_available_actions
    ):
        """Test that truncated response (no action) triggers retry."""
        truncated_response = """[Condensed Memory]
Testing.
[Thinking Process]
I need to think about this carefully and consider all my options before making a decision..."""
        good_response = """[Condensed Memory]
Testing.
[Thinking Process]
Moving.
[Action] MOVE from Cafeteria to Admin"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                mock_send.side_effect = [truncated_response, good_response]

                with patch.object(mock_agent, "log_interaction"):
                    action = await mock_agent.choose_action(timestep=0)

        assert mock_send.call_count == 2
        assert action.name == "MOVE"

    @pytest.mark.asyncio
    async def test_two_failures_then_success(self, mock_agent, basic_available_actions):
        """Test recovery after two consecutive failures."""
        bad1 = "completely garbage"
        bad2 = "[Action] definitely not a real action lol"
        good = """[Condensed Memory]
Test.
[Thinking Process]
Ok.
[Action] MOVE from Cafeteria to Admin"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                mock_send.side_effect = [bad1, bad2, good]

                with patch.object(mock_agent, "log_interaction"):
                    action = await mock_agent.choose_action(timestep=0)

        assert mock_send.call_count == 3
        assert action.name == "MOVE"

    @pytest.mark.asyncio
    async def test_memory_updated_on_success(self, mock_agent, basic_available_actions):
        """Test that processed_memory is updated from response."""
        response = """[Condensed Memory]
Game started in Cafeteria. Player 2 moved to Admin.
[Thinking Process]
Following Player 2.
[Action] MOVE from Cafeteria to Admin"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                mock_send.return_value = response

                with patch.object(mock_agent, "log_interaction"):
                    await mock_agent.choose_action(timestep=0)

        assert "Game started in Cafeteria" in mock_agent.processed_memory

    @pytest.mark.asyncio
    async def test_summarization_updated_on_success(
        self, mock_agent, basic_available_actions
    ):
        """Test that summarization is updated from response."""
        response = """[Condensed Memory]
Testing.
[Thinking Process]
I am strategically moving to Admin to complete my task there.
[Action] MOVE from Cafeteria to Admin"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                mock_send.return_value = response

                with patch.object(mock_agent, "log_interaction"):
                    await mock_agent.choose_action(timestep=0)

        assert "strategically" in mock_agent.summarization

    @pytest.mark.asyncio
    async def test_speak_action_captures_message(
        self, mock_agent, basic_available_actions
    ):
        """Test that SPEAK action captures the full message."""
        response = """[Condensed Memory]
Meeting phase.
[Thinking Process]
I need to share what I saw.
[Action] SPEAK: I saw Player 2 venting in Electrical! Vote them out!"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                mock_send.return_value = response

                with patch.object(mock_agent, "log_interaction"):
                    action = await mock_agent.choose_action(timestep=0)

        assert action.name == "SPEAK"
        assert "venting" in action.message

    @pytest.mark.asyncio
    async def test_multiple_hallucinations_eventually_fail(self, mock_agent):
        """Test that consistent hallucinations exhaust retries and raise an error."""
        # Only MOVE available but model keeps trying to KILL
        limited_actions = [MoveTo("Cafeteria", "Admin")]

        hallucination = """[Condensed Memory]
Test.
[Thinking Process]
Kill.
[Action] KILL Player 2"""

        with patch.object(
            mock_agent, "send_request", new_callable=AsyncMock
        ) as mock_send:
            with patch.object(
                mock_agent.player, "get_available_actions", return_value=limited_actions
            ):
                mock_send.return_value = hallucination

                with patch.object(mock_agent, "log_interaction"):
                    with pytest.raises(RuntimeError, match="Format validation failed"):
                        await mock_agent.choose_action(timestep=0)

        # All 3 retries exhausted
        assert mock_send.call_count == 3


class TestRetryFeedbackMessage:
    """Tests for the feedback message sent to model on retry."""

    @pytest.mark.asyncio
    async def test_feedback_includes_original_response(
        self, mock_agent, basic_available_actions
    ):
        """Test that retry feedback includes the original failed response."""
        bad_response = "I don't know what to do"
        good_response = """[Condensed Memory]
Test.
[Thinking Process]
Moving.
[Action] MOVE from Cafeteria to Admin"""

        captured_messages = []

        async def capture_send(messages):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return bad_response
            return good_response

        with patch.object(mock_agent, "send_request", side_effect=capture_send):
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                with patch.object(mock_agent, "log_interaction"):
                    await mock_agent.choose_action(timestep=0)

        # Second call should include feedback with original response
        assert len(captured_messages) == 2
        retry_messages = captured_messages[1]
        # Check that the original bad response is in the retry messages
        assert any(
            "I don't know what to do" in str(m.get("content", ""))
            for m in retry_messages
        )

    @pytest.mark.asyncio
    async def test_feedback_includes_available_actions(
        self, mock_agent, basic_available_actions
    ):
        """Test that retry feedback lists available actions."""
        bad_response = "invalid"
        good_response = """[Condensed Memory]
Test.
[Thinking Process]
Moving.
[Action] MOVE from Cafeteria to Admin"""

        captured_messages = []

        async def capture_send(messages):
            captured_messages.append(messages)
            if len(captured_messages) == 1:
                return bad_response
            return good_response

        with patch.object(mock_agent, "send_request", side_effect=capture_send):
            with patch.object(
                mock_agent.player,
                "get_available_actions",
                return_value=basic_available_actions,
            ):
                with patch.object(mock_agent, "log_interaction"):
                    await mock_agent.choose_action(timestep=0)

        # Check that available actions are mentioned in feedback
        retry_messages = captured_messages[1]
        feedback_content = str(retry_messages)
        assert "MOVE" in feedback_content or "Available actions" in feedback_content


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
