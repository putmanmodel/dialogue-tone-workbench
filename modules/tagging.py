from __future__ import annotations

import re
from typing import Dict


POLARITY_BY_TONE = {
    "defensive": "negative",
    "apologetic": "neutral",
    "passive-aggressive": "negative",
    "avoidant": "neutral",
    "sincere": "positive",
    "escalating": "negative",
}

BASE_INTENSITY_BY_TONE = {
    "defensive": 0.46,
    "apologetic": 0.38,
    "passive-aggressive": 0.67,
    "avoidant": 0.29,
    "sincere": 0.33,
    "escalating": 0.82,
}


def tag_variant(text: str, target_tone: str) -> Dict[str, object]:
    detected_tone = detect_tone(text=text, target_tone=target_tone)
    intensity = estimate_intensity(text=text, target_tone=target_tone, detected_tone=detected_tone)
    polarity = infer_polarity(detected_tone)

    return {
        "target_tone": target_tone,
        "detected_tone": detected_tone.upper().replace("-", "_"),
        "intensity": intensity,
        "polarity": polarity,
        "notes": "",
    }


def detect_tone(text: str, target_tone: str) -> str:
    lowered = text.lower()

    if "sorry" in lowered:
        return "apologetic"

    defensive_markers = [
        "i'm pushing back on that",
        "i am pushing back on that",
        "i'm pushing back on it",
        "i am pushing back on it",
        "i'm pushing back on that version of it",
        "i am pushing back on that version of it",
        "that does not mean i'm avoiding this forever",
        "that does not mean i am avoiding this forever",
        "not turn this into something worse",
        "do not turn it into something else",
        "not going to act like that is fine",
        "that does not mean i agree",
        "not brush it off",
        "not because i'm ignoring the problem",
        "not because i am ignoring the problem",
        "not dismiss you",
        "trying to steady this",
    ]
    if any(marker in lowered for marker in defensive_markers):
        return "defensive"

    ordered_patterns = [
        (
            "avoidant",
            [
                "i don't want to keep",
                "i do not want to keep",
                "i'd rather leave it there",
                "i'd rather leave it alone",
                "i would rather leave it there",
                "let's leave it there",
                "step away from it",
                "i do not want to drag this out",
                "we do not need to turn it into more than that",
                "i'm not going to push this any further",
                "i am not going to push this any further",
                "that is really all i meant",
                "before this gets worse between us",
                "i need a minute with it",
                "i need to step away from it",
                "i'm letting it go for now",
                "i am letting it go for now",
                "i'm leaving it there for now",
                "i am leaving it there for now",
                "i'm stepping back from it now",
                "i am stepping back from it now",
                "i'd rather not keep pushing on it",
                "i can see the issue",
                "let's come back to it later",
                "i need a minute before we keep doing this",
                "i think we need to pause this conversation for now",
                "i do not want to keep arguing over a misquote",
                "i need to step back before we keep going",
            ],
        ),
        (
            "defensive",
            [
                "that wasn't the point i was making",
                "that was not the point i was making",
                "that wasn't how i meant it",
                "that was not how i meant it",
                "i'm not trying to take a shot at you",
                "i am not trying to take a shot at you",
                "i'm not trying to overstate it",
                "i am not trying to overstate it",
                "i need to be able to say",
                "i need to be able to say that i can't deal with this right now",
                "i'm pushing back on that",
                "i am pushing back on that",
                "i'm not going to act like this feels fair",
                "i am not going to act like this feels fair",
                "so i'm not going to act like it was nothing",
                "so i am not going to act like it was nothing",
                "that does not mean i agree",
                "i'm pushing back on that version of it",
                "i am pushing back on that version of it",
                "i'm not going to act like that is fine",
                "i am not going to act like that is fine",
            ],
        ),
        (
            "apologetic",
            [
                "sorry",
                "i'm sorry",
                "i am sorry",
                "not easy to hear",
                "i can hear how that landed",
                "if that sounded",
                "came out wrong",
                "i can see that came out wrong",
                "i know this is hard to hear",
            ],
        ),
        (
            "sincere",
            [
                "what i was trying to say",
                "i meant something a little different",
                "i really think",
                "i honestly think",
                "i'm trying to put this plainly",
                "i am trying to put this plainly",
                "i need you to hear that",
                "i'm asking a simple timing question",
                "i'm only trying to check whether this is coming through",
                "i am only trying to check whether this is coming through",
                "be clear with you",
                "to be precise",
                "that landed more painfully than you may realize",
                "that had more impact on me than you may realize",
                "i'm being honest about how this feels",
                "i am being honest about how this feels",
                "i don't feel heard here",
                "we can move on from it",
                "i'm willing to let this rest",
                "i am willing to let this rest",
                "i'm willing to leave it there",
                "i am willing to leave it there",
                "we can leave it there",
                "we can move on from it",
                "that's not what i'm saying",
                "that is not what i'm saying",
                "this has moved beyond what i agreed to",
                "this does not match the agreement as i understood it",
            ],
        ),
        (
            "passive-aggressive",
            [
                "apparently",
                "sure, whatever",
                "sure, let's act like",
                "clearly there is no point",
                "whole thing",
                "whole production",
                "too much now",
                "let me just ignore my own limit",
            ],
        ),
        (
            "escalating",
            [
                "going nowhere",
                "i'm done repeating myself",
                "i am done repeating myself",
                "stop pushing past that",
                "we are wasting time",
                "can we stop making it weird",
                "i am not going to argue about it",
                "i'm done here",
                "this should not be this hard",
                "do not put words in my mouth",
                "please do not attribute that position to me",
                "i'm not going to keep defending words i did not say",
                "i am not going to keep defending words i did not say",
                "i am not going along with it",
                "i'm not accepting that shift",
                "i am not accepting that shift",
            ],
        ),
    ]

    for tone, patterns in ordered_patterns:
        if any(pattern in lowered for pattern in patterns):
            return tone

    return target_tone


def estimate_intensity(text: str, target_tone: str, detected_tone: str) -> float:
    score = BASE_INTENSITY_BY_TONE.get(detected_tone, BASE_INTENSITY_BY_TONE.get(target_tone, 0.4))
    lowered = text.lower()

    amplifiers = ["never", "always", "apparently", "whole thing", "wasting time", "done here", "fight", "going nowhere"]
    softeners = ["sorry", "trying to", "i think", "i really think", "i honestly think", "let's", "care", "hear that", "be clear with you"]

    score += 0.06 * sum(token in lowered for token in amplifiers)
    score -= 0.04 * sum(token in lowered for token in softeners)

    if "!" in text:
        score += 0.08

    score = max(0.0, min(1.0, score))
    return round(score, 2)


def infer_polarity(detected_tone: str) -> str:
    return POLARITY_BY_TONE.get(detected_tone, "neutral")


def build_context_flags(context: str) -> Dict[str, bool]:
    lowered = (context or "").strip().lower()
    tokens = set(re.findall(r"[a-z0-9']+", lowered))
    return {
        "has_context": bool(lowered),
        "tense": any(token in lowered for token in ["argument", "fight", "tense", "conflict", "strained", "frustrated", "escalat"]),
        "formal": any(token in lowered for token in ["meeting", "work", "office", "professional"]),
        "urgent": bool(tokens & {"urgent", "deadline", "tonight", "immediately", "asap"}) or "right away" in lowered or "right now" in lowered,
        "relational": any(token in lowered for token in ["friend", "partner", "boss", "coworker", "parent", "family", "team"]),
        "misunderstanding": any(token in lowered for token in ["misunder", "misread", "misquot", "wrong way"]),
    }
