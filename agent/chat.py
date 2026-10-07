from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .prompts.initial import initial_prompt
from .action_controller import SpotifyController

# ponytail: counts messages, not tokens; use a token counter if replies get long
MAX_HISTORY = 20


class AssistantModel:
    def __init__(self, model="qwen3:8b", llm=None):
        self._spotify = None  # built on first music tool call, not at startup
        self.messages = []
        tools = self._spotify_tools()
        llm = (llm or ChatOllama(model=model, reasoning=False)).bind_tools(tools)

        def call_model(state: MessagesState):
            # Refresh the clock every turn - this process stays up for days
            system = SystemMessage(f"The current time is {datetime.now():%H:%M:%S}.\n\n{initial_prompt}")
            return {"messages": [llm.invoke([system, *state["messages"]])]}

        # model -> (tool calls? -> tools -> model) -> reply
        graph = StateGraph(MessagesState)
        graph.add_node("model", call_model)
        graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))  # errors go back to the model
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", tools_condition)
        graph.add_edge("tools", "model")
        self.graph = graph.compile()

    def spotify(self):
        if self._spotify is None:
            self._spotify = SpotifyController()
        return self._spotify

    def respond(self, user_msg):
        """Run one turn, including any tool calls, and return NIA's reply"""
        messages = self.graph.invoke({"messages": [*self.messages, HumanMessage(user_msg)]})["messages"]
        # Keep whole turns only, so a tool result never loses the call it answers
        self.messages = trim_messages(messages, max_tokens=MAX_HISTORY, token_counter=len,
                                      strategy="last", start_on="human")
        return messages[-1].content

    def _spotify_tools(self):
        sp = self.spotify  # called inside each tool so the client stays lazy

        @tool
        def now_playing() -> str:
            """What is playing on Spotify right now, and whether it is paused"""
            return sp().now_playing()

        @tool
        def play(query: str, kind: Literal["track", "playlist", "album"] = "track") -> str:
            """Search Spotify and play the top hit. query is the name, plus the artist if known.
            kind is 'album' or 'playlist' when the user says so, otherwise 'track'"""
            return {"track": sp().play_track, "playlist": sp().play_playlist, "album": sp().play_album}[kind](query)

        @tool
        def add_to_queue(query: str) -> str:
            """Queue a track to play next. query is the track name, plus the artist if known"""
            return sp().add_to_queue(query)

        @tool
        def pause() -> str:
            """Pause Spotify"""
            return sp().pause()

        @tool
        def resume() -> str:
            """Resume Spotify playback"""
            return sp().resume()

        @tool
        def skip(direction: Literal["next", "previous"] = "next") -> str:
            """Skip to the next track, or go back to the previous one"""
            return sp().next() if direction == "next" else sp().previous()

        @tool
        def set_volume(percent: int) -> str:
            """Set Spotify volume to an exact percent, 0-100"""
            return sp().set_volume(percent)

        @tool
        def change_volume(step: int) -> str:
            """Turn Spotify volume up (positive step) or down (negative step) by step percent. Use 10 or -10 unless told an amount"""
            return sp().change_volume(step)

        @tool
        def shuffle(on: bool = True) -> str:
            """Turn Spotify shuffle on or off"""
            return sp().shuffle(on)

        @tool
        def repeat(mode: Literal["track", "context", "off"]) -> str:
            """Repeat the current track, repeat the current playlist or album ('context'), or turn repeat off"""
            return sp().repeat(mode)

        return [now_playing, play, add_to_queue, pause, resume, skip, set_volume, change_volume, shuffle, repeat]
