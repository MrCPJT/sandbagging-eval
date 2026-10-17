from behaviours.base import Behaviour


def get_behaviour(name: str) -> Behaviour:
    if name == "sandbagging":
        from behaviours.sandbagging import BEHAVIOUR

        return BEHAVIOUR
    raise ValueError(f"unknown behaviour: {name}")
