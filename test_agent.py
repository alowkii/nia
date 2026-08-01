"""Offline checks for the agent's JSON parsing and action dispatch. Run: python test_agent.py"""
from agent.chat import AssistantModel
from agent.action_controller import SpotifyController


class FakeSpotify(SpotifyController):
    """Records calls instead of hitting the Spotify API."""
    def __init__(self):
        self.calls = []

    def __getattribute__(self, name):
        if name in ("calls", "__class__", "__dict__"):
            return object.__getattribute__(self, name)
        def record(*args):
            self.calls.append((name, args))
            return f"did {name}"
        return record


def test_clean_json():
    a = AssistantModel()
    assert a.clean_json('{"a": 1}') == {"a": 1}
    assert a.clean_json('sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    for bad in ("", "   ", "no json here"):
        try:
            a.clean_json(bad)
            assert False, f"should have raised on {bad!r}"
        except ValueError:
            pass


def test_dispatch():
    a = AssistantModel()
    a._spotify = FakeSpotify()

    def act(subtype, keyword="none"):
        return a.take_action({"action_type": "music", "action_platform": "Spotify",
                              "action_subtype": subtype, "action_keyword": keyword})

    assert act("play_track", "Highway to Hell") == "did play_track"
    assert a._spotify.calls[-1] == ("play_track", ("Highway to Hell",))

    # every subtype the prompt advertises must route somewhere
    import json
    subtypes = json.load(open("agent/actions.json"))["sub_actions"]["music"]
    for subtype in subtypes:
        assert act(subtype) is not None, f"{subtype} has no handler"

    # repeat_* must pass the right mode through, not just fire
    act("repeat_off")
    assert a._spotify.calls[-1] == ("repeat", ("off",))

    # non-music and non-spotify are not our problem
    assert act("nonexistent_subtype") is None
    assert a.take_action({"action_type": "weather"}) is None
    assert a.take_action({"action_type": "music", "action_platform": "youtube",
                          "action_subtype": "play_track"}) is None

    # dispatch must not build a Spotify client for actions it does not own
    fresh = AssistantModel()
    assert fresh.take_action({"action_type": "weather"}) is None
    assert fresh._spotify is None, "built a Spotify client for a non-music action"


if __name__ == "__main__":
    test_clean_json()
    test_dispatch()
    print("ok")
