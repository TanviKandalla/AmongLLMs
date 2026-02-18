"""
Test that Impostor agents are told who their fellow Impostors are.

Regression test for a bug where list_of_impostors was always [] because
it was populated in the same loop that created agents, so the list was
still empty at agent construction time.
"""

import os
import re
import tempfile

import pytest

os.environ["EXPERIMENT_PATH"] = tempfile.gettempdir()

from amongagents.envs.game import AmongUs
from amongagents.envs.configs.game_config import SEVEN_MEMBER_GAME
from amongagents.envs.configs.agent_config import ALL_RANDOM, ALL_LLM


def test_impostor_agents_know_teammates():
    """Every Impostor agent's system prompt must list all Impostor names."""
    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    # There should be exactly num_impostors names in the list
    assert len(game.list_of_impostors) == SEVEN_MEMBER_GAME["num_impostors"]

    # Every name in the list should belong to an actual Impostor player
    impostor_player_names = {p.name for p in game.players if p.identity == "Impostor"}
    assert set(game.list_of_impostors) == impostor_player_names


def test_impostor_system_prompt_contains_teammates():
    """LLM Impostor agents must have teammate names in their system prompt.
    
    NOTE: This is a loose sanity check. Other tests verify proper placement
    and formatting. This just ensures the names appear somewhere.
    """
    from amongagents.envs.configs.agent_config import ALL_LLM

    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    impostor_names = game.list_of_impostors
    assert len(impostor_names) == 2, "Expected 2 impostors in 7-player game"

    # Get all impostor agents
    impostor_agents = [
        agent for agent, player in zip(game.agents, game.players)
        if player.identity == "Impostor"
    ]
    
    for agent in impostor_agents:
        # All impostor names should appear somewhere in the prompt
        # (More specific tests verify WHERE and HOW they appear)
        for name in impostor_names:
            assert name in agent.system_prompt, (
                f"Impostor agent's system prompt is missing teammate '{name}'"
            )


def test_impostor_list_populated_in_prompt():
    """Impostor prompt must contain teammates in natural language format."""
    from amongagents.envs.configs.agent_config import ALL_LLM

    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    impostor_names = game.list_of_impostors
    for agent, player in zip(game.agents, game.players):
        if player.identity != "Impostor":
            continue
        
        # Look for natural language format
        prompt = agent.system_prompt
        
        # Should have "YOUR FELLOW IMPOSTOR(S):" header
        assert "YOUR FELLOW IMPOSTOR(S):" in prompt or "You are the ONLY Impostor" in prompt, (
            f"Impostor agent {player.name}'s prompt missing teammate section"
        )
        
        # Each OTHER impostor name should appear (not the player's own name in the list)
        teammates = [name for name in impostor_names if name != player.name]
        for teammate in teammates:
            assert teammate in prompt, (
                f"Impostor agent {player.name}'s prompt is missing teammate '{teammate}'"
            )


def test_impostor_list_before_examples():
    """Impostor list must appear BEFORE examples, not after.
    
    Models pay more attention to information that comes before examples.
    Tacking the list at the end (after "DO NOT pick...") is terrible UX.
    """
    from amongagents.envs.configs.agent_config import ALL_LLM

    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    for agent, player in zip(game.agents, game.players):
        if player.identity != "Impostor":
            continue
        
        prompt = agent.system_prompt
        
        # Find where the impostor list and examples are mentioned
        list_idx = prompt.find("YOUR FELLOW IMPOSTOR(S):")
        if list_idx == -1:  # Handle solo impostor case
            list_idx = prompt.find("You are the ONLY Impostor")
        example_idx = prompt.find("[Condensed Memory]")  # Start of examples
        
        assert list_idx != -1, f"No impostor teammate info found in {player.name}'s prompt"
        assert example_idx != -1, f"No examples found in {player.name}'s prompt"
        
        # The list should come BEFORE the examples start, not after
        assert list_idx < example_idx, (
            f"FAIL: Impostor list appears at position {list_idx}, "
            f"but examples start at {example_idx}. "
            f"List should be in the main instructions, not tacked on at the end!"
        )


def test_impostor_list_natural_language_format():
    """Impostor list should be natural language, not Python syntax.
    
    Bad format: "List of impostors: ['Player 1: yellow', 'Player 7: lime']"
    Good format: "YOUR FELLOW IMPOSTOR(S): Player 1: yellow, Player 7: lime"
    """
    from amongagents.envs.configs.agent_config import ALL_LLM

    game = AmongUs(
        game_config=SEVEN_MEMBER_GAME,
        agent_config=ALL_LLM,
    )
    game.initialize_game()

    for agent, player in zip(game.agents, game.players):
        if player.identity != "Impostor":
            continue
        
        prompt = agent.system_prompt
        
        # Should NOT have Python list syntax
        assert not re.search(r"\['.*?'\]", prompt), (
            f"FAIL: {player.name}'s prompt uses Python list syntax ['name1', 'name2']. "
            f"Use natural language instead."
        )
        
        # SHOULD have natural language format
        assert ("YOUR FELLOW IMPOSTOR(S):" in prompt or 
                "You are the ONLY Impostor" in prompt), (
            f"FAIL: {player.name}'s prompt missing natural language teammate section"
        )
