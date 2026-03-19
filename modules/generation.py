from __future__ import annotations

import re
from typing import Dict, List


VARIANT_CATEGORIES = [
    "defensive",
    "apologetic",
    "passive-aggressive",
    "avoidant",
    "sincere",
    "escalating",
]

GENERATION_STRATEGY = "utterance_type_paraphrase_rules_v1"
UTTERANCE_TYPES = [
    "clarification",
    "apology",
    "accusation",
    "concern",
    "vulnerability",
    "request_or_question",
    "pause_or_deescalation",
    "denial_or_correction",
    "objection_or_mismatch",
    "refusal_or_boundary",
    "reassurance",
    "dismissal_or_detachment",
    "generic_statement",
]


def normalize_line(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def generate_variants(
    base_line: str,
    speaker: str = "",
    target: str = "",
    context: str = "",
) -> Dict[str, object]:
    clean_line = normalize_line(base_line)
    framing = build_framing(speaker=speaker, target=target, context=context)
    utterance_type = detect_utterance_type(clean_line, framing)
    semantic_frame = build_semantic_frame(clean_line, utterance_type, framing)

    variants = []
    for tone in VARIANT_CATEGORIES:
        options = build_tone_options(utterance_type, tone, semantic_frame, framing)
        text = select_template(clean_line, f"{utterance_type}:{tone}", options)
        variants.append({"text": text, "target_tone": tone})

    return {
        "utterance_type": utterance_type,
        "generation_strategy": GENERATION_STRATEGY,
        "variants": variants,
    }


def build_framing(speaker: str = "", target: str = "", context: str = "") -> Dict[str, object]:
    target_name = (target or "").strip()
    speaker_name = (speaker or "").strip()
    context_text = (context or "").strip().lower()
    context_tokens = set(tokenize(context_text))

    relationship_words = ["friend", "partner", "boss", "coworker", "parent", "family", "team", "relationship"]
    urgency_words = {"urgent", "deadline", "tonight", "immediately", "asap"}

    return {
        "address": f", {target_name}" if target_name else "",
        "target_name": target_name,
        "speaker": speaker_name,
        "is_tense": any(word in context_text for word in ["argument", "fight", "tense", "conflict", "strained", "frustrated", "escalat"]),
        "is_formal": any(word in context_text for word in ["meeting", "work", "office", "professional"]),
        "is_urgent": bool(context_tokens & urgency_words) or "right away" in context_text or "right now" in context_text,
        "is_relational": any(word in context_text for word in relationship_words),
        "mentions_misunderstanding": any(word in context_text for word in ["misunder", "misread", "misquot", "wrong way"]),
        "context_hint": context.strip(),
    }


def detect_utterance_type(line: str, framing: Dict[str, object]) -> str:
    lowered = line.lower().replace("’", "'")
    tokens = tokenize(lowered)

    if contains_any(lowered, ["sorry", "apologize", "apologies", "my bad"]):
        return "apology"

    if contains_any(
        lowered,
        [
            "it should be okay",
            "it'll be okay",
            "it will be okay",
            "we'll find a solution",
            "we will find a solution",
            "i'm sure we'll find a solution",
            "i am sure we'll find a solution",
            "i’m sure we’ll find a solution",
        ],
    ):
        return "reassurance"

    if contains_any(
        lowered,
        [
            "i need a minute",
            "i need a second",
            "give me a minute",
            "give me a second",
            "let's not do this right now",
            "let's not go there",
            "let us not do this right now",
            "not do this right now",
            "talk about this later",
            "come back to this later",
            "not get into this right now",
            "not go there",
        ],
    ):
        return "pause_or_deescalation"

    if contains_any(
        lowered,
        [
            "i'm not saying that",
            "i am not saying that",
            "that's not what i'm saying",
            "that is not what i'm saying",
            "don't put that on me",
            "do not put that on me",
            "don't twist my words",
            "do not twist my words",
        ],
    ):
        return "denial_or_correction"

    if contains_any(
        lowered,
        [
            "this isn't what i agreed to",
            "this is not what i agreed to",
            "this wasn't what i agreed to",
            "that isn't what i agreed to",
            "that is not what i agreed to",
            "not what i agreed to",
            "not what we agreed to",
            "not part of what i agreed to",
            "not what i signed up for",
        ],
    ):
        return "objection_or_mismatch"

    if starts_with_any(lowered, ["you're not", "you are not", "you aren't"]):
        return "accusation"

    if contains_any(
        lowered,
        [
            "you took that the wrong way",
            "you twisted",
            "you made it",
            "you keep making",
            "you heard that",
            "you heard me",
            "you read that",
            "not listening to me",
            "not hearing me",
        ],
    ):
        return "accusation"

    if contains_any(
        lowered,
        [
            "it stings",
            "stings more each time",
            "can't stop thinking about this",
            "cannot stop thinking about this",
            "can't shake it",
            "cannot shake it",
            "weighing on me",
            "stuck with me",
        ],
    ):
        return "vulnerability"

    if contains_any(
        lowered,
        [
            "that hurt",
            "that actually hurt",
            "that hurt me",
            "that upset me",
            "that stung",
            "i feel ",
            "i feel hurt",
            "i feel dismissed",
            "i feel ignored",
        ],
    ):
        return "vulnerability"

    if lowered.startswith("that wasn't") or lowered.startswith("that's not") or lowered.startswith("that isn't"):
        return "clarification"

    if lowered.startswith("i meant") or lowered.startswith("i'm trying to explain") or lowered.startswith("i am trying to explain"):
        return "clarification"

    if contains_any(
        lowered,
        [
            "didn't mean",
            "wasn't my point",
            "that's not my point",
            "not my point",
            "that wasn't what i meant",
            "not what i meant",
            "trying to explain",
            "trying to explain myself",
            "trying to clarify",
            "wrong way",
            "you heard",
            "you read",
            "misunderstood",
        ],
    ):
        return "clarification"

    if lowered.endswith("?") or contains_any(
        lowered,
        [
            "does this work",
            "what time is it",
            "can you",
            "could you",
            "would you",
            "i'm asking",
            "checking whether",
            "checking if",
        ],
    ):
        return "request_or_question"

    if contains_any(
        lowered,
        [
            "can't deal",
            "cannot deal",
            "can't do this",
            "don't want to",
            "do not want to",
            "leave me alone",
            "stop",
            "i'm done",
            "back off",
        ],
    ):
        return "refusal_or_boundary"

    if contains_any(
        lowered,
        [
            "this might be a problem",
            "this could be a problem",
            "this is a problem",
            "might be a problem",
            "i think this is a problem",
            "i think this might be a problem",
            "need to address",
            "need to deal",
            "should address",
            "should deal",
            "should maybe check",
            "check on that again",
            "dangerous territory",
            "red flags",
            "waving little red flags",
            "worried",
            "concerned",
            "might need",
            "soon",
        ],
    ):
        return "concern"

    if contains_any(
        lowered,
        [
            "don't worry",
            "it's okay",
            "it is okay",
            "we'll be okay",
            "we will be okay",
            "you're okay",
            "you are okay",
            "it's fine",
            "i'm sure we'll",
            "i am sure we'll",
            "i’m sure we’ll",
            "it should be okay",
            "we'll find a solution",
            "we will find a solution",
            "we’ll find a solution",
        ],
    ):
        return "reassurance"

    if contains_any(
        lowered,
        [
            "that's how it usually plays out",
            "that is how it usually plays out",
            "that's how this usually goes",
            "that is how this usually goes",
            "it usually goes this way",
            "that's how it goes",
        ],
    ):
        return "dismissal_or_detachment"

    if contains_any(lowered, ["sure, whatever", "sure.", "sure", "whatever", "fine.", "fine", "okay.", "okay", "alright", "leave it"]):
        return "dismissal_or_detachment"

    if tokens and tokens[0] == "you" and contains_any(lowered, ["wrong", "always", "never", "took", "ignored", "twisted"]):
        return "accusation"

    return "generic_statement"


def build_semantic_frame(line: str, utterance_type: str, framing: Dict[str, object]) -> Dict[str, str]:
    lower = line.lower()
    address = str(framing.get("address", ""))

    frame = {
        "line": strip_terminal_punctuation(line),
        "line_lower": strip_terminal_punctuation(lower),
        "address": address,
        "subject": "this",
        "request_kind": "generic",
        "detail": "I'm saying this plainly",
        "soft_detail": "I'm trying to say this clearly",
        "firm_detail": "I'm being direct about this",
    }

    if utterance_type == "clarification":
        if "point" in lower:
            frame["detail"] = "that wasn't the point I was making"
            frame["soft_detail"] = "I was trying to make a different point"
            frame["firm_detail"] = "you're missing the point I was making"
        elif "explain" in lower:
            frame["detail"] = "I'm trying to explain where I'm coming from"
            frame["soft_detail"] = "I'm trying to explain myself more clearly"
            frame["firm_detail"] = "I'm trying to explain myself, not argue"
        elif "wrong way" in lower or "misunder" in lower:
            frame["detail"] = "it landed differently than I meant it to"
            frame["soft_detail"] = "I meant it differently from how it landed"
            frame["firm_detail"] = "that's not how I meant it"
        else:
            frame["detail"] = "that wasn't how I meant it"
            frame["soft_detail"] = "I meant something a little different"
            frame["firm_detail"] = "that's not what I was saying"

        if framing.get("is_relational"):
            frame["soft_detail"] = "I'm trying to be clear with you about what I meant"
        if framing.get("is_tense"):
            frame["firm_detail"] = "we're already tense, and that's still not what I meant"
        if framing.get("is_formal"):
            frame["detail"] = frame["detail"].replace("wasn't", "was not").replace("that's", "that is")
            frame["soft_detail"] = formalize(frame["soft_detail"])
            frame["firm_detail"] = formalize(frame["firm_detail"])

    elif utterance_type == "apology":
        frame["detail"] = "I'm sorry for how that came across"
        frame["soft_detail"] = "I know that landed badly"
        frame["firm_detail"] = "I'm owning that I said it badly"

    elif utterance_type == "accusation":
        if "listening" in lower or "hearing me" in lower:
            frame["detail"] = "you're not really hearing what I'm saying"
            frame["soft_detail"] = "I don't feel heard here"
            frame["firm_detail"] = "you're not listening to what I'm actually saying"
        elif "wrong way" in lower:
            frame["detail"] = "I think you read that more harshly than I meant it"
            frame["soft_detail"] = "I think you took that differently than I intended"
            frame["firm_detail"] = "you're making this harsher than it was"
        else:
            frame["detail"] = "this feels unfairly put on me"
            frame["soft_detail"] = "I think you're reading me in the worst possible way"
            frame["firm_detail"] = "you're turning this into something it wasn't"

        if framing.get("is_relational"):
            frame["soft_detail"] = "I don't feel heard by you right now" if "listening" in lower or "hearing" in lower else frame["soft_detail"]
        if framing.get("is_formal"):
            frame["soft_detail"] = formalize(frame["soft_detail"])
            frame["firm_detail"] = formalize(frame["firm_detail"])

    elif utterance_type == "concern":
        if "dangerous territory" in lower:
            frame["detail"] = "this is getting into dangerous territory"
            frame["soft_detail"] = "I do not think this stays harmless if we keep going"
            frame["firm_detail"] = "we are getting too close to something risky"
        elif "red flags" in lower or "instinct" in lower:
            frame["detail"] = "my instincts are telling me something is off"
            frame["soft_detail"] = "this is setting off red flags for me"
            frame["firm_detail"] = "too many red flags are showing up to ignore"
        elif "address" in lower or "deal" in lower:
            frame["detail"] = "we should deal with this soon"
            frame["soft_detail"] = "I think this needs attention sooner rather than later"
            frame["firm_detail"] = "this needs to be dealt with soon"
        elif "problem" in lower:
            frame["detail"] = "this might be a problem"
            frame["soft_detail"] = "I think this could turn into a problem"
            frame["firm_detail"] = "this is becoming a problem"
        else:
            frame["detail"] = "this needs attention"
            frame["soft_detail"] = "I think we should take this seriously"
            frame["firm_detail"] = "we should not keep putting this off"

        if framing.get("is_urgent"):
            frame["detail"] = "we need to deal with this soon"
            frame["soft_detail"] = "I think this needs attention right away"
            frame["firm_detail"] = "this needs to be addressed right now"
        if framing.get("is_formal"):
            frame["soft_detail"] = formalize(frame["soft_detail"])
            frame["firm_detail"] = "this requires attention soon"
        if framing.get("is_relational"):
            frame["soft_detail"] = "I think we need to deal with this before it gets worse between us"

    elif utterance_type == "vulnerability":
        if "hurt" in lower:
            frame["detail"] = "that actually hurt"
            frame["soft_detail"] = "that landed more painfully than you may realize"
            frame["firm_detail"] = "that hurt, and I'm not brushing it off"
        elif "upset" in lower:
            frame["detail"] = "that upset me"
            frame["soft_detail"] = "that hit me harder than I expected"
            frame["firm_detail"] = "that upset me, and it matters"
        elif "i feel" in lower:
            frame["detail"] = "I'm telling you how this is landing on me"
            frame["soft_detail"] = "I'm being honest about how this feels"
            frame["firm_detail"] = "this is affecting me, and I'm not minimizing that"
        else:
            frame["detail"] = "this hit me harder than I expected"
            frame["soft_detail"] = "I'm being honest that this affected me"
            frame["firm_detail"] = "this affected me, and I'm not minimizing it"

        if framing.get("is_relational"):
            frame["soft_detail"] = "that hurt, especially coming from you"
        if framing.get("is_formal"):
            frame["soft_detail"] = "That had more impact on me than you may realize"
            frame["firm_detail"] = formalize(frame["firm_detail"])

    elif utterance_type == "request_or_question":
        if "what time" in lower:
            frame["request_kind"] = "time"
            frame["detail"] = "I'm just asking what time it is"
            frame["soft_detail"] = "I'm asking a simple timing question"
            frame["firm_detail"] = "I'm asking a straightforward question about the time"
        elif "work" in lower or "working" in lower or "coming through" in lower or "check" in lower:
            frame["request_kind"] = "check"
            frame["detail"] = "I'm just checking whether this is working"
            frame["soft_detail"] = "I'm only trying to check whether this is coming through"
            frame["firm_detail"] = "I'm asking for a simple check here"
        else:
            frame["detail"] = "I'm asking a straightforward question"
            frame["soft_detail"] = "I'm just trying to ask this plainly"
            frame["firm_detail"] = "I'm asking directly"

        if framing.get("is_formal"):
            frame["soft_detail"] = formalize(frame["soft_detail"])
            frame["firm_detail"] = "I am asking directly for a clear answer"

    elif utterance_type == "pause_or_deescalation":
        if "minute" in lower or "second" in lower:
            frame["detail"] = "I need a minute"
            frame["soft_detail"] = "I need a minute before we keep doing this"
            frame["firm_detail"] = "I need you to stop pushing this right now and give me a minute"
        elif "go there" in lower:
            frame["detail"] = "let's not go there"
            frame["soft_detail"] = "I don't think we need to go there right now"
            frame["firm_detail"] = "we are not going down that road"
        else:
            frame["detail"] = "let's not do this right now"
            frame["soft_detail"] = "I think we need to pause this conversation for now"
            frame["firm_detail"] = "we are not doing this right now"

        if framing.get("is_relational"):
            frame["soft_detail"] = "I need a little space before this gets worse between us"
        if framing.get("is_tense"):
            frame["firm_detail"] = "we are too heated for this right now, and I need it to stop"
        if framing.get("is_formal"):
            frame["soft_detail"] = formalize(frame["soft_detail"])
            frame["firm_detail"] = formalize(frame["firm_detail"])

    elif utterance_type == "denial_or_correction":
        if "not saying" in lower or "what i'm saying" in lower or "what i am saying" in lower:
            frame["detail"] = "I'm not saying that"
            frame["soft_detail"] = "that's not what I'm saying"
            frame["firm_detail"] = "do not put words in my mouth"
        else:
            frame["detail"] = "that's not what I meant"
            frame["soft_detail"] = "I need to correct that reading"
            frame["firm_detail"] = "you're putting words in my mouth"

        if framing.get("mentions_misunderstanding"):
            frame["soft_detail"] = "I need to correct that because that is not what I said"
        if framing.get("is_formal"):
            frame["detail"] = frame["detail"].replace("I'm", "I am").replace("that's", "that is")
            frame["soft_detail"] = formalize(frame["soft_detail"])
            frame["firm_detail"] = "please do not attribute that position to me"

    elif utterance_type == "objection_or_mismatch":
        if "agreed" in lower:
            frame["detail"] = "this isn't what I agreed to"
            frame["soft_detail"] = "this has moved beyond what I agreed to"
            frame["firm_detail"] = "this is outside what I agreed to, and I'm not accepting that shift"
        else:
            frame["detail"] = "this is not what I signed up for"
            frame["soft_detail"] = "this is not the arrangement I agreed to"
            frame["firm_detail"] = "this changed from what was agreed, and I'm not going along with that"

        if framing.get("is_formal"):
            frame["detail"] = frame["detail"].replace("isn't", "is not")
            frame["soft_detail"] = formalize(frame["soft_detail"])
            frame["firm_detail"] = formalize(frame["firm_detail"])
        if framing.get("is_relational"):
            frame["soft_detail"] = "this is not what we agreed to, and that matters to me"

    elif utterance_type == "refusal_or_boundary":
        if "right now" in lower:
            frame["detail"] = "I can't deal with this right now"
            frame["soft_detail"] = "I need some space from this right now"
            frame["firm_detail"] = "this is not something I'm doing right now"
        else:
            frame["detail"] = "I need to step back from this"
            frame["soft_detail"] = "I don't want to keep doing this"
            frame["firm_detail"] = "I'm drawing a line here"

        if framing.get("is_relational"):
            frame["soft_detail"] = "I need a little space from this right now"
        if framing.get("is_urgent"):
            frame["firm_detail"] = "this is not something I can keep doing right now"
        if framing.get("is_formal"):
            frame["soft_detail"] = formalize(frame["soft_detail"])
            frame["firm_detail"] = formalize(frame["firm_detail"])

    elif utterance_type == "reassurance":
        if "solution" in lower or "i'm sure we'll" in lower or "i am sure we'll" in lower or "i’m sure we’ll" in lower:
            frame["detail"] = "I'm sure we'll find a solution"
            frame["soft_detail"] = "we can work our way through this"
            frame["firm_detail"] = "we are going to find a way through this"
        elif "it should be okay" in lower or "it'll be okay" in lower or "it will be okay" in lower:
            frame["detail"] = "it should be okay"
            frame["soft_detail"] = "I really think this is going to be okay"
            frame["firm_detail"] = "this does not need to turn into a worst-case scenario"
        else:
            frame["detail"] = "I'm trying to calm this down"
            frame["soft_detail"] = "I'm saying this is okay"
            frame["firm_detail"] = "I'm telling you this does not need to spiral"

    elif utterance_type == "dismissal_or_detachment":
        if "usually plays out" in lower or "usually goes" in lower:
            frame["detail"] = "that's how it usually plays out"
            frame["soft_detail"] = "this is the pattern I've come to expect"
            frame["firm_detail"] = "I'm tired of watching the same pattern repeat"
        elif lower.startswith("sure"):
            frame["detail"] = "I'm going along with it, but I'm checked out"
            frame["soft_detail"] = "I'm leaving it there for now"
            frame["firm_detail"] = "I'm done arguing about it"
        elif lower.startswith("fine") or lower.startswith("okay"):
            frame["detail"] = "I'm dropping it for now"
            frame["soft_detail"] = "I'm stepping back from it now"
            frame["firm_detail"] = "I'm done arguing this point"
        else:
            frame["detail"] = "I'm backing away from this"
            frame["soft_detail"] = "I'm not investing more into this"
            frame["firm_detail"] = "I'm done engaging with this"

        if framing.get("is_relational"):
            frame["soft_detail"] = "I'm dropping it before this gets worse between us"
        if framing.get("is_formal"):
            frame["soft_detail"] = formalize(frame["soft_detail"])
            frame["firm_detail"] = formalize(frame["firm_detail"])

    else:
        frame["detail"] = frame["line"]
        frame["soft_detail"] = "I'm trying to put this plainly"
        frame["firm_detail"] = "I'm saying this directly"

    if framing.get("is_formal"):
        frame["soft_detail"] = formalize(frame["soft_detail"])
    if framing.get("is_urgent"):
        frame["firm_detail"] = add_urgency(frame["firm_detail"])

    return frame


def build_tone_options(
    utterance_type: str,
    tone: str,
    frame: Dict[str, str],
    framing: Dict[str, object],
) -> List[str]:
    builders = {
        "clarification": clarification_options,
        "apology": apology_options,
        "accusation": accusation_options,
        "concern": concern_options,
        "vulnerability": vulnerability_options,
        "request_or_question": request_or_question_options,
        "pause_or_deescalation": pause_or_deescalation_options,
        "denial_or_correction": denial_or_correction_options,
        "objection_or_mismatch": objection_or_mismatch_options,
        "refusal_or_boundary": refusal_or_boundary_options,
        "reassurance": reassurance_options,
        "dismissal_or_detachment": dismissal_or_detachment_options,
        "generic_statement": generic_statement_options,
    }
    return builders.get(utterance_type, generic_statement_options)(tone, frame, framing)


def clarification_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    relational_soft = "I can see that came out wrong." if framing.get("is_relational") else None
    formal_soft = "To be precise, that was not what I meant." if framing.get("is_formal") else None
    options = {
        "defensive": [
            f"{sentence_case(detail)}{address}.",
            f"{sentence_case(soft)}{address}; I'm not trying to take a shot at you.",
        ],
        "apologetic": [
            f"Sorry{address}. {sentence_case(soft)}.",
            f"I'm sorry{address}. {sentence_case(detail)}.",
        ],
        "passive-aggressive": [
            f"So me clarifying it still somehow is not enough{address}.",
            f"Of course every clarification from me has to become its own issue{address}.",
        ],
        "avoidant": [
            f"{sentence_case(soft)}{address}, but I don't want to keep going in circles.",
            f"{sentence_case(detail)}{address}, and I would rather leave it there.",
        ],
        "sincere": [
            f"{sentence_case(soft)}{address}.",
            f"What I was trying to say was a little different{address}.",
        ],
        "escalating": [
            f"If I have to keep restating it, this is going nowhere{address}.",
            f"{sentence_case(firm)}{address}, and I'm done repeating myself.",
        ],
    }
    if relational_soft:
        options["apologetic"].append(f"Sorry{address}. {relational_soft}")
        options["sincere"].append(f"I'm trying to be clear with you about what I meant{address}.")
    if formal_soft:
        options["defensive"].append(f"{formal_soft[:-1]}{address}.")
        options["sincere"].append(f"{formal_soft[:-1]}{address}.")
        options["apologetic"].append(f"Sorry{address}. To be precise, that was not what I meant.")
    return options[tone]


def apology_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    relational_tail = "I care more about fixing it than defending it." if framing.get("is_relational") else "I should have handled that better."
    options = {
        "defensive": [
            f"I'm sorry{address}, but I was not trying to be hurtful.",
            f"{detail.capitalize()}{address}, even if that was not my intention.",
        ],
        "apologetic": [
            f"{detail.capitalize()}{address}.",
            f"I'm sorry{address}. {soft.capitalize()}.",
        ],
        "passive-aggressive": [
            f"Well, sorry{address}. Apparently even that came out wrong.",
            f"I said it badly, clearly, since we're still here{address}.",
        ],
        "avoidant": [
            f"I'm sorry{address}. Can we leave it there for now?",
            f"{detail.capitalize()}{address}, and I do not want to drag this out.",
        ],
        "sincere": [
            f"{detail.capitalize()}{address}. {relational_tail}",
            f"I'm sorry{address}. I can hear how that landed.",
        ],
        "escalating": [
            f"I said I'm sorry{address}, and I'm not going to keep groveling over it.",
            f"{firm.capitalize()}{address}, but this does not need to become a bigger fight.",
        ],
    }
    return options[tone]


def accusation_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    options = {
        "defensive": [
            f"{soft.capitalize()}{address}, and I'm pushing back on that.",
            f"This does not feel fair to me{address}.",
        ],
        "apologetic": [
            f"Sorry{address}, but {soft}.",
            f"I'm not trying to start another fight{address}. I just think {detail}.",
        ],
        "passive-aggressive": [
            f"Sure, let's act like {firm}{address}.",
            f"So I'm supposed to keep talking like I'm being heard{address}.",
        ],
        "avoidant": [
            f"I think {detail}{address}, but I don't want to keep sparring over it.",
            f"{soft.capitalize()}{address}, and I'm stepping back before this gets worse.",
        ],
        "sincere": [
            f"I honestly think {soft}{address}.",
            f"From where I'm standing, {detail}{address}.",
        ],
        "escalating": [
            f"{firm.capitalize()}{address}, and I'm tired of acting like that is not happening.",
            f"If this keeps getting twisted, we're not getting anywhere{address}.",
        ],
    }
    return options[tone]


def vulnerability_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    options = {
        "defensive": [
            f"{sentence_case(detail)}{address}, and I'm not treating it like nothing.",
            f"{sentence_case(firm)}{address}.",
        ],
        "apologetic": [
            f"Sorry{address}. I don't want to make this bigger, but {detail}.",
            f"I'm sorry{address}. {sentence_case(soft)}.",
        ],
        "passive-aggressive": [
            f"So I'm supposed to pretend {detail}{address} did not matter.",
            f"Sure, let's act like that was nothing{address}.",
        ],
        "avoidant": [
            f"{sentence_case(detail)}{address}, but I don't want to drag this out right now.",
            f"{sentence_case(soft)}{address}, and I need a minute with it.",
        ],
        "sincere": [
            f"{sentence_case(detail)}{address}.",
            f"{sentence_case(soft)}{address}.",
        ],
        "escalating": [
            f"{sentence_case(detail)}{address}, and I'm done pretending otherwise.",
            f"That hurt{address}, and I'm done swallowing it.",
        ],
    }
    return options[tone]


def concern_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    urgency = "We do not have much time on this." if framing.get("is_urgent") else ""
    options = {
        "defensive": [
            f"I'm not trying to overstate it{address}. {sentence_case(detail)}.",
            f"{sentence_case(soft)}{address}, not because I'm trying to make a scene.",
        ],
        "apologetic": [
            f"Sorry{address}, but {detail}.",
            f"I know this is not easy to hear{address}. {sentence_case(soft)}.",
        ],
        "passive-aggressive": [
            f"Sure, let's keep pretending {firm}{address}.",
            f"Pointing out that {detail} is suddenly too much now{address}.",
        ],
        "avoidant": [
            f"I still think {detail}{address}, but I do not want to keep pressing it right now.",
            f"I can see the issue{address}, but I'd rather not keep pushing on it. {urgency}".strip(),
        ],
        "sincere": [
            f"{sentence_case(soft)}{address}.",
            f"I really think {detail}{address}.",
        ],
        "escalating": [
            f"{sentence_case(firm)}{address}, and we are wasting time pretending otherwise.",
            f"If we keep dodging it, this is only getting worse{address}.",
        ],
    }
    if framing.get("is_urgent"):
        options["passive-aggressive"].append(f"Sure, let's wait until this is urgent enough to count{address}.")
        options["escalating"].append(f"This needs attention right now{address}, and we're wasting time pretending otherwise.")
    return [clean_spacing(option) for option in options[tone]]


def request_or_question_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    request_kind = frame["request_kind"]

    defensive_first = "I'm just trying to check on this."
    apologetic_first = f"Sorry{address}, I'm just trying to check on this."
    sincere_second = f"{sentence_case(detail)}{address}."

    if request_kind == "time":
        defensive_first = f"I'm only asking what time it is{address}."
        apologetic_first = f"Sorry{address}, I'm just asking what time it is."
        sincere_second = f"{sentence_case(detail)}{address}."
    elif request_kind == "check":
        defensive_first = f"I'm just trying to check whether this is working{address}."
        apologetic_first = f"Sorry{address}, I'm just trying to check whether this is working."
        sincere_second = f"{sentence_case(soft)}{address}."

    options = {
        "defensive": [
            defensive_first,
            f"{sentence_case(soft)}{address}, not trying to make this bigger.",
        ],
        "apologetic": [
            apologetic_first,
            f"I'm sorry{address}. {sentence_case(soft)}.",
        ],
        "passive-aggressive": [
            f"I'm just asking a simple question, which apparently is a problem{address}.",
            f"Somehow a straightforward question has to become a whole thing{address}.",
        ],
        "avoidant": [
            f"{sentence_case(soft)}{address}, and that is really all this is.",
            f"It's a simple question{address}. We do not need to turn it into more than that.",
        ],
        "sincere": [
            f"{sentence_case(soft)}{address}.",
            sincere_second,
        ],
        "escalating": [
            f"I'm asking a simple question, and this should not be this hard{address}.",
            f"{sentence_case(firm)}{address}, so can we stop making it weird?",
        ],
    }
    if request_kind == "time":
        options["passive-aggressive"].append(f"I'm asking what time it is{address}, not opening a debate.")
    if request_kind == "check":
        options["passive-aggressive"].append(f"I'm just trying to check whether this is working{address}, and somehow it turned into a whole production.")
    return options[tone]


def pause_or_deescalation_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    options = {
        "defensive": [
            f"{sentence_case(detail)}{address}, and that does not mean I'm avoiding this forever.",
            f"{sentence_case(soft)}{address}, not turn this into something worse.",
        ],
        "apologetic": [
            f"Sorry{address}. {sentence_case(soft)}.",
            f"I'm sorry{address}, but {detail}.",
        ],
        "passive-aggressive": [
            f"Sure, let's keep pushing until nobody has any patience left{address}.",
            f"Asking for a minute is somehow too much now{address}.",
        ],
        "avoidant": [
            f"{sentence_case(soft)}{address}. Let's come back to it later.",
            f"{sentence_case(detail)}{address}, and I'm stepping back before this gets worse.",
        ],
        "sincere": [
            f"{sentence_case(detail)}{address}.",
            f"{sentence_case(soft)}{address}.",
        ],
        "escalating": [
            f"{sentence_case(firm)}{address}.",
            f"I said {detail}{address}. Leave it there for now.",
        ],
    }
    if framing.get("is_tense"):
        options["sincere"].append(f"We are both getting too frustrated for this right now{address}.")
        options["avoidant"].append(f"This is getting too heated{address}, so I'm stepping back.")
    return [clean_spacing(option) for option in options[tone]]


def denial_or_correction_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    options = {
        "defensive": [
            f"{sentence_case(detail)}{address}, so do not turn it into something else.",
            f"{sentence_case(soft)}{address}, and I'm pushing back on that version of it.",
        ],
        "apologetic": [
            f"Sorry{address}, but {soft}.",
            f"I'm sorry{address} if this is getting tense, but {detail}.",
        ],
        "passive-aggressive": [
            f"So now you get to decide what I said{address}.",
            f"Sure, let's pretend I said something I did not{address}.",
        ],
        "avoidant": [
            f"{sentence_case(soft)}{address}, and I do not want to keep arguing over a misquote.",
            f"{sentence_case(detail)}{address}. I'm leaving it there.",
        ],
        "sincere": [
            f"{sentence_case(detail)}{address}.",
            f"{sentence_case(soft)}{address}.",
        ],
        "escalating": [
            f"{sentence_case(firm)}{address}.",
            f"I'm correcting that now{address}, and I'm not defending words I did not say.",
        ],
    }
    if framing.get("mentions_misunderstanding"):
        options["sincere"].append(f"I need to correct that because it is not what I meant{address}.")
    return [clean_spacing(option) for option in options[tone]]


def objection_or_mismatch_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    options = {
        "defensive": [
            f"{sentence_case(detail)}{address}, and I'm not treating that like it's fine.",
            f"{sentence_case(soft)}{address}, so I'm pushing back on that change.",
        ],
        "apologetic": [
            f"Sorry{address}, but {detail}.",
            f"I'm sorry{address}. {sentence_case(soft)}.",
        ],
        "passive-aggressive": [
            f"Sure, let's just rewrite the agreement in real time{address}.",
            f"What I agreed to apparently only matters when it is convenient{address}.",
        ],
        "avoidant": [
            f"{sentence_case(soft)}{address}, and I do not want to keep forcing it right now.",
            f"{sentence_case(detail)}{address}. I need to step back before we keep going.",
        ],
        "sincere": [
            f"{sentence_case(detail)}{address}.",
            f"{sentence_case(soft)}{address}.",
        ],
        "escalating": [
            f"{sentence_case(firm)}{address}.",
            f"This is not what I agreed to{address}, and I am not going along with it.",
        ],
    }
    if framing.get("is_formal"):
        options["sincere"].append(f"This does not match the agreement as I understood it{address}.")
        options["defensive"].append(f"This has shifted from what was agreed{address}, and I need to say that plainly.")
    return [clean_spacing(option) for option in options[tone]]


def refusal_or_boundary_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    options = {
        "defensive": [
            f"I need to be able to say {detail}{address}.",
            f"{sentence_case(soft)}{address}, and that is not unreasonable.",
        ],
        "apologetic": [
            f"I'm sorry{address}, but {detail}.",
            f"Sorry{address}. {sentence_case(soft)}.",
        ],
        "passive-aggressive": [
            f"Even saying {detail} is somehow too much now{address}.",
            f"Sure, let me just ignore my own limit{address}.",
        ],
        "avoidant": [
            f"{soft.capitalize()}{address}. Let's leave it there.",
            f"{detail.capitalize()}{address}, and I need to step away from it.",
        ],
        "sincere": [
            f"{sentence_case(detail)}{address}.",
            f"{sentence_case(soft)}{address}, and I need you to hear that.",
        ],
        "escalating": [
            f"{sentence_case(firm)}{address}, and I am not going to argue about it.",
            f"I said {detail}{address}. Stop pushing past that.",
        ],
    }
    if framing.get("is_urgent"):
        options["sincere"].append(f"{sentence_case(detail)}{address}, and I do not have the bandwidth for more right now.")
        options["escalating"].append(f"I cannot keep doing this right now{address}, and I am not going to fight about that.")
    return options[tone]


def reassurance_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    options = {
        "defensive": [
            f"{sentence_case(soft)}{address}, and I'm trying to steady this, not brush it off.",
            f"{sentence_case(detail)}{address}, not because I'm ignoring the problem.",
        ],
        "apologetic": [
            f"Sorry{address}. {sentence_case(soft)}.",
            f"I'm sorry if that sounded dismissive{address}. {sentence_case(detail)}.",
        ],
        "passive-aggressive": [
            f"I was trying to reassure you{address}, but apparently even that is wrong.",
            f"So {firm}{address} is somehow offensive too.",
        ],
        "avoidant": [
            f"{sentence_case(soft)}{address}, and I do not want to spiral over it.",
            f"{sentence_case(detail)}{address}. Let's leave it there for now.",
        ],
        "sincere": [
            f"{sentence_case(detail)}{address}.",
            f"{sentence_case(soft)}{address}.",
        ],
        "escalating": [
            f"{sentence_case(firm)}{address}, and I need us to stop feeding the panic.",
            f"I'm trying to settle this down{address}, not let it spin further.",
        ],
    }
    return options[tone]


def dismissal_or_detachment_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    detail = frame["detail"]
    soft = frame["soft_detail"]
    firm = frame["firm_detail"]
    address = frame["address"]
    if "usually plays out" in detail or "pattern i've come to expect" in soft:
        options = {
            "defensive": [
                f"{sentence_case(soft)}{address}, and I'm not pretending otherwise.",
                f"{sentence_case(detail)}{address}, so forgive me for not sounding surprised.",
            ],
            "apologetic": [
                f"Sorry{address}. {sentence_case(soft)}.",
                f"I'm sorry if that sounds bleak{address}. {sentence_case(detail)}.",
            ],
            "passive-aggressive": [
                f"Of course this is how it usually plays out{address}.",
                f"Right, the same pattern again, because apparently that is all we do{address}.",
            ],
            "avoidant": [
                f"{sentence_case(soft)}{address}, and I do not have the energy to chase it again.",
                f"{sentence_case(detail)}{address}. I'd rather leave it there.",
            ],
            "sincere": [
                f"{sentence_case(detail)}{address}.",
                f"{sentence_case(soft)}{address}.",
            ],
            "escalating": [
                f"{sentence_case(firm)}{address}.",
                f"I'm done acting surprised by the same pattern{address}.",
            ],
        }
        return options[tone]

    options = {
        "defensive": [
            f"I'm backing off because {detail}{address}, not because you are right.",
            f"I'm stepping back here{address}, and that does not mean I agree.",
        ],
        "apologetic": [
            f"Sorry{address}. I'm dropping it.",
            f"I'm sorry if that sounded cold{address}. {soft.capitalize()}.",
        ],
        "passive-aggressive": [
            f"Fine{address}. Clearly there is no point saying more.",
            f"Sure, whatever{address}. Let's act like that solves it.",
        ],
        "avoidant": [
            f"{soft.capitalize()}{address}.",
            f"{detail.capitalize()}{address}, and I'd rather leave it alone.",
        ],
        "sincere": [
            f"Okay{address}. We can move on from it.",
            f"We can leave it there{address}.",
        ],
        "escalating": [
            f"{firm.capitalize()}{address}.",
            f"Fine{address}. I'm done talking about it.",
        ],
    }
    if framing.get("is_relational"):
        options["avoidant"].append(f"I'm dropping it before this gets worse between us{address}.")
        options["passive-aggressive"].append(f"Sure, let's just shut it down and pretend that fixes us{address}.")
    return options[tone]


def generic_statement_options(tone: str, frame: Dict[str, str], framing: Dict[str, object]) -> List[str]:
    address = frame["address"]
    options = {
        "defensive": [
            f"I'm being direct here{address}, not trying to turn it into more than that.",
            f"That's the point I'm trying to make{address}.",
        ],
        "apologetic": [
            f"Sorry{address}. I was trying to say that more clearly.",
            f"I'm sorry{address}. I could have put that better.",
        ],
        "passive-aggressive": [
            f"So even saying this plainly is too much{address}.",
            f"That straightforward point is somehow the issue now{address}.",
        ],
        "avoidant": [
            f"I've said what I mean{address}, and I do not want to keep unpacking it.",
            f"That is really all I was trying to say{address}.",
        ],
        "sincere": [
            f"I'm trying to put this plainly{address}.",
            f"I'm trying to say this clearly{address}.",
        ],
        "escalating": [
            f"I'm being direct, and I am not going to keep dressing it up{address}.",
            f"If even that simple point becomes a fight, we are wasting time{address}.",
        ],
    }
    return options[tone]


def select_template(base_line: str, key: str, options: List[str]) -> str:
    seed_value = sum(ord(char) for char in f"{base_line}:{key}")
    return options[seed_value % len(options)]


def strip_terminal_punctuation(text: str) -> str:
    return text.rstrip(" .!?")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def starts_with_any(text: str, prefixes: list[str]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def formalize(text: str) -> str:
    return (
        text.replace("I'm", "I am")
        .replace("don't", "do not")
        .replace("can't", "cannot")
        .replace("it's", "it is")
        .replace("that's", "that is")
    )


def add_urgency(text: str) -> str:
    if "soon" in text or "right now" in text or "immediately" in text:
        return text
    if "conversation" in text or "going nowhere" in text or "fight" in text or "done here" in text:
        return text
    return f"{text} soon"


def clean_spacing(text: str) -> str:
    return " ".join(text.split())


def sentence_case(text: str) -> str:
    cleaned = re.sub(r"\bi\b", "I", text.strip())
    if not cleaned:
        return cleaned
    return cleaned[0].upper() + cleaned[1:]
