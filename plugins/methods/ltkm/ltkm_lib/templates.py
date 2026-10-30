"""Persona-aware question templates, keyed by DNLI relation label.

The paper's own worked examples ("I like playing basketball." -> relation
"like_sports" -> "What sports do you like?"; "My parents live in
Australia." -> relation "live_place" -> "What place do your parents live?")
both follow one generic pattern: `relation = "{r1}_{r2}"` (verb_noun) ->
`"What {r2} do {e1} {r1}?"`. The paper states this generic template is the
*fallback* ("for persona sentences not in DNLI, we can use a relation
extraction model... to achieve this goal"), implying curated templates
exist for DNLI's own fixed relation set -- but the paper never publishes
that per-relation table, so `RELATION_TEMPLATES` below is this repo's own
reconstruction, hand-written to read naturally per relation (using the
generic r1_r2 pattern as a starting point, adjusted where it would read
awkwardly, e.g. "physical_attribute" -> "have_pet"-style genitive phrasing
instead of a literal "What pet do you have?" reading for every relation).
The relation name list itself (75+ names) is confirmed from Welleck et al.
2019 Appendix A.1, not guessed.
"""
from __future__ import annotations

# {e1} is filled with the (already pronoun-flipped, see question_construction.py)
# subject phrase, e.g. "you" or "your parents".
RELATION_TEMPLATES: dict[str, str] = {
    "place_origin": "Where are {e1} originally from?",
    "live_in_citystatecountry": "Where do {e1} live?",
    "live_in_general": "Where do {e1} live?",
    "nationality": "What is {e1} nationality?",
    "employed_by_company": "Where do {e1} work?",
    "employed_by_general": "What do {e1} do for work?",
    "has_profession": "What is {e1} profession?",
    "previous_profession": "What did {e1} used to do for work?",
    "job_status": "What is {e1} job status?",
    "teach": "What do {e1} teach?",
    "school_status": "What is {e1} school status?",
    "has_degree": "What degree do {e1} have?",
    "attend_school": "What school do {e1} attend?",
    "like_general": "What do {e1} like?",
    "like_food": "What food do {e1} like?",
    "like_drink": "What drink do {e1} like?",
    "like_animal": "What animal do {e1} like?",
    "like_movie": "What movie do {e1} like?",
    "like_music": "What music do {e1} like?",
    "like_read": "What do {e1} like to read?",
    "like_sports": "What sports do {e1} like?",
    "like_watching": "What do {e1} like watching?",
    "like_activity": "What activity do {e1} like?",
    "like_goto": "Where do {e1} like to go?",
    "dislike": "What do {e1} dislike?",
    "has_hobby": "What is {e1} hobby?",
    "has_ability": "What are {e1} able to do?",
    "member_of": "What group are {e1} a member of?",
    "want_do": "What do {e1} want to do?",
    "want_job": "What job do {e1} want?",
    "want": "What do {e1} want?",
    "favorite_food": "What is {e1} favorite food?",
    "favorite_color": "What is {e1} favorite color?",
    "favorite_book": "What is {e1} favorite book?",
    "favorite_movie": "What is {e1} favorite movie?",
    "favorite_music": "What is {e1} favorite music?",
    "favorite_music_artist": "Who is {e1} favorite music artist?",
    "favorite_activity": "What is {e1} favorite activity?",
    "favorite_drink": "What is {e1} favorite drink?",
    "favorite_show": "What is {e1} favorite show?",
    "favorite_place": "What is {e1} favorite place?",
    "favorite_hobby": "What is {e1} favorite hobby?",
    "favorite_season": "What is {e1} favorite season?",
    "favorite_animal": "What is {e1} favorite animal?",
    "favorite_sport": "What is {e1} favorite sport?",
    "favorite": "What is {e1} favorite thing?",
    "own": "What do {e1} own?",
    "have": "What do {e1} have?",
    "not_have": "What do {e1} not have?",
    "have_pet": "What pet do {e1} have?",
    "have_sibling": "What siblings do {e1} have?",
    "have_children": "Do {e1} have children?",
    "have_family": "What is {e1} family like?",
    "have_vehicle": "What vehicle do {e1} have?",
    "physical_attribute": "What is {e1} physical attribute?",
    "misc_attribute": "What else is true about {e1}?",
    "has_age": "How old are {e1}?",
    "marital_status": "What is {e1} marital status?",
    "gender": "What is {e1} gender?",
    "other": "What is true about {e1}?",
}


def build_question(subject_phrase: str, relation: str) -> str:
    """Build a persona-aware question for a subject phrase (already
    pronoun-flipped to 2nd person, e.g. "you"/"your parents") and a DNLI
    relation label. Falls back to the paper's own generic "What {r2} do
    {e1} {r1}?" pattern for any relation not in `RELATION_TEMPLATES`
    (covers both genuinely unknown labels and any future DNLI relation
    additions this table hasn't been updated for)."""
    template = RELATION_TEMPLATES.get(relation)
    if template is not None:
        return template.format(e1=subject_phrase)

    if "_" in relation:
        r1, r2 = relation.split("_", 1)
        return f"What {r2.replace('_', ' ')} do {subject_phrase} {r1}?"

    return f"What is {subject_phrase} {relation}?"
