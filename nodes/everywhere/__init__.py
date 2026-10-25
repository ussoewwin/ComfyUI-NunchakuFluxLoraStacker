from .use_everywhere import (
    hswq_SeedEverywhere,
    hswq_AnythingEverywhere,
    hswq_AnythingSomewhere,
    hswq_AnythingEverywhereTriplet,
    hswq_SimpleString,
    hswq_ComboClone,
    hswq_AnythingEverywherePrompts,
)

UE_VERSION = "7.8"

NODE_CLASS_MAPPINGS = {
    "hswq_AnythingEverywhere": hswq_AnythingEverywhere,
    "hswq_AnythingSomewhere": hswq_AnythingSomewhere,
    "hswq_AnythingEverywhereTriplet": hswq_AnythingEverywhereTriplet,
    "hswq_SimpleString": hswq_SimpleString,
    "hswq_ComboClone": hswq_ComboClone,
    "hswq_SeedEverywhere": hswq_SeedEverywhere,
    "hswq_AnythingEverywherePrompts": hswq_AnythingEverywherePrompts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "hswq_AnythingEverywhere": "hswq_AnythingEverywhere",
    "hswq_AnythingSomewhere": "hswq_AnythingSomewhere",
    "hswq_AnythingEverywhereTriplet": "hswq_AnythingEverywhereTriplet",
    "hswq_SimpleString": "hswq_SimpleString",
    "hswq_ComboClone": "hswq_ComboClone",
    "hswq_SeedEverywhere": "hswq_SeedEverywhere",
    "hswq_AnythingEverywherePrompts": "hswq_AnythingEverywherePrompts",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "UE_VERSION"]
