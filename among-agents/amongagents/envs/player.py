from amongagents.envs.action import (
    COMMON_ACTIONS,
    CREWMATE_ACTIONS,
    IMPOSTER_ACTIONS,
    CompleteTask,
)

PLAYER_COLORS = [
    "red",
    "blue",
    "green",
    "pink",
    "orange",
    "yellow",
    "black",
    "white",
    "purple",
    "brown",
    "cyan",
    "lime",
]


class Player:
    def __init__(self, name, identity, color, personality, location=None):
        # Basic player information
        self.name = f"{name}: {color}"
        self.color = color
        self.identity = identity  # e.g., "Crewmate" or "Imposter"
        self.location = location  # Initially, the player might not have a location
        self.personality = personality

        # Player history
        self.observation_history = []
        self.action_history = []
        self.location_info = None
        self.observations_since_last_action = 0

        # Player options
        self.COMMON_ACTIONS = COMMON_ACTIONS
        self.SPECIAL_ACTIONS = []
        self.available_actions = []

        # Player status
        self.is_alive = True
        self.tasks = []
        self.reported_death = False

    def __repr__(self) -> str:
        return f"{self.name} ({self.identity})"

    def __str__(self) -> str:
        return self.__repr__()

    def assign_tasks(self, tasks):
        self.tasks = tasks
        for task in tasks:
            task.assign_to(self)

    def get_all_actions(self):
        """Return the actions that the player can take."""
        return self.COMMON_ACTIONS + self.SPECIAL_ACTIONS

    def set_available_actions(self, actions):
        self.available_actions = actions

    def get_available_actions(self):
        if self.is_alive:
            return self.available_actions
        else:
            return []

    def make_action(self, env, action, choose_location="Cafeteria"):
        if action.name == "ViewMonitor":
            action.execute(env, self, choose_location)
        else:
            action.execute(env, self)
        if env.current_phase == "task":
            record = {
                "timestep": env.timestep,
                "phase": env.current_phase,
                "action": action,
            }
        elif env.current_phase == "meeting":
            round = env.game_config["discussion_rounds"] - env.discussion_rounds_left
            record = {
                "timestep": env.timestep,
                "phase": env.current_phase,
                "round": round,
                "action": action,
            }
        self.action_history.append(record)
        # Reset counter when player takes an action
        self.observations_since_last_action = 0

    def receive(self, message, info_type):
        if info_type == "location":
            self.location_info = message
        elif info_type == "action":
            self.observation_history.append(message)
            self.observations_since_last_action += 1

    def location_info_prompt(self):
        text = self.location_info
        return text

    def available_actions_prompt(self):
        text = "Available actions:\n"
        for i, action in enumerate(self.available_actions):
            text += f"{i + 1}. {action}\n"
        return text

    def action_history_prompt(self):
        """Return recent action history. Older actions should be captured in condensed memory."""
        text = "Action history:\n"
        if len(self.action_history) == 0:
            text += "No actions have been taken yet.\n"
        else:
            recent_actions = self.action_history[-4:]
            for i, record in enumerate(recent_actions):
                timestep = record["timestep"]
                current_phase = record["phase"]
                action = record["action"]
                if current_phase == "task":
                    if type(action) == CompleteTask:
                        action_text = str(action)
                    else:
                        action_text = action.action_text()
                    text += (
                        f"Timestep {timestep}: [{current_phase} phase] {action_text}\n"
                    )
                elif current_phase == "meeting":
                    round = record["round"]
                    text += f"Timestep {timestep}: [{current_phase} phase - round {round}] {action.action_text()}\n"
        text += "\n"
        return text

    def observation_history_prompt(self):
        # Show all observations since player's last action
        text = "Observation history:\n"
        if (
            len(self.observation_history) == 0
            or self.observations_since_last_action == 0
        ):
            text += "No observations have been made yet.\n"
        else:
            for i, message in enumerate(
                self.observation_history[-self.observations_since_last_action :]
            ):
                text += f"{i + 1}. {message}\n"
        text += "\n"
        return text

    def tasks_prompt(self):
        text = "Your Assigned Tasks:\n"
        if len(self.tasks) == 0:
            text += "No tasks have been assigned yet.\n"
        else:
            for i, task in enumerate(self.tasks):
                completed = "[completed]" if task.check_completion() else ""
                text += f"{i + 1}. {task} {completed}\n"
                path = task.find_path(self.location, identity=self.identity)
                if len(path) > 1:
                    path = "->".join(path)
                else:
                    path = "You are already at the task location."
                text += f"Path: {path}\n"
        text += "\n"
        return text

    def all_info_prompt(self):
        text = self.location_info_prompt()
        text += self.observation_history_prompt()
        text += self.action_history_prompt()
        text += self.tasks_prompt()
        text += self.available_actions_prompt()
        return text


class Crewmate(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, identity="Crewmate")
        self.SPECIAL_ACTIONS = CREWMATE_ACTIONS


class Impostor(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, identity="Impostor")
        self.SPECIAL_ACTIONS = IMPOSTER_ACTIONS
        self.kill_cooldown = 0

    def available_actions_prompt(self):
        text = ""
        if self.kill_cooldown > 0:
            text += f"KILL is on cooldown ({self.kill_cooldown} timestep{'s' if self.kill_cooldown != 1 else ''} remaining).\n"
        text += super().available_actions_prompt()
        return text
