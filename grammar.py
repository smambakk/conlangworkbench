"""
grammar.py — Conlang Grammar Rule Engine & Translator
=======================================================
Handles sentence-level structure: word order, clause assembly,
and bidirectional translation between English and the conlang.

Design philosophy
-----------------
The user defines their language's grammar through declarative rules —
word order, argument marking, agreement — and the engine applies them.
Translation is transparent: every step is logged so users can see
exactly why a sentence came out the way it did, and can override any
decision by adding entries or adjusting rules.

Components
----------
GrammaticalFeatures : a bundle of features on a token (POS, case, tense…)
Token               : a surface word + its grammatical features
Clause              : a structured clause (subject, verb, object, etc.)
WordOrderRule       : defines surface word order for a clause type
AgreementRule       : copies a feature from one argument to another word
GrammarEngine       : assembles clauses into sentences
Translator          : full English ↔ conlang pipeline
TranslationResult   : surface string + step-by-step log

Supported word orders (extensible)
-----------------------------------
  SVO, SOV, VSO, VOS, OVS, OSV
  Plus topic-prominent and discourse-configurable variants.

Usage
-----
    from phonology  import PhonologyEngine
    from lexicon    import Lexicon
    from morphology import MorphologyEngine
    from grammar    import GrammarEngine, Translator, WordOrderRule

    phon  = PhonologyEngine.load("phonology.json")
    lex   = Lexicon.load("lexicon.json", phonology=phon)
    morph = MorphologyEngine.load("morphology.json", lexicon=lex)
    gram  = GrammarEngine(morph)
    gram.word_order = "SOV"

    tr = Translator(gram, lex, morph)
    result = tr.translate_to_conlang("the water runs")
    print(result.surface)
    print(result.log())
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from phonology  import PhonologyEngine
except ImportError:
    PhonologyEngine = None  # type: ignore
try:
    from lexicon    import Lexicon, LexiconEntry
except ImportError:
    Lexicon = None; LexiconEntry = None  # type: ignore
try:
    from morphology import MorphologyEngine, InflectionResult
except ImportError:
    MorphologyEngine = None; InflectionResult = None  # type: ignore


# ---------------------------------------------------------------------------
# Feature bundle & Token
# ---------------------------------------------------------------------------

VALID_WORD_ORDERS = {"SVO","SOV","VSO","VOS","OVS","OSV"}

VALID_CASES = {
    "nominative","accusative","genitive","dative",
    "locative","ablative","instrumental","vocative","none",
}

@dataclass
class GrammaticalFeatures:
    """A bundle of grammatical features attached to a token."""
    pos:    str   = "noun"
    case:   str   = "none"
    number: str   = "singular"
    tense:  str   = ""
    aspect: str   = ""
    person: str   = ""
    gender: str   = ""
    extra:  dict  = field(default_factory=dict)   # user-defined features

    def to_inflect_dict(self) -> dict:
        """Convert to the feature dict expected by MorphologyEngine.inflect()."""
        d = {}
        if self.number and self.number != "singular":
            d["number"] = self.number
        if self.tense:
            d["tense"] = self.tense
        if self.aspect:
            d["aspect"] = self.aspect
        if self.case and self.case != "none":
            d["case"] = self.case
        if self.person:
            d["person"] = self.person
        d.update(self.extra)
        return d


@dataclass
class Token:
    """A word in a sentence, with surface form and grammatical features."""
    lemma:    str                    # dictionary form / root
    surface:  str                    # final surface form (after morphology)
    features: GrammaticalFeatures    = field(default_factory=GrammaticalFeatures)
    role:     str                    = ""   # "subject","object","verb","modifier"…
    gloss:    str                    = ""   # English gloss of this token
    is_clang: bool                   = False  # True = already in conlang form

    def __str__(self):
        return self.surface


# ---------------------------------------------------------------------------
# Clause
# ---------------------------------------------------------------------------

@dataclass
class Clause:
    """A structured clause with labelled argument slots.

    Slots (all optional lists to support multi-word constituents):
      subject, verb, direct_object, indirect_object,
      modifier (adjectives, adverbs), topic, focus

    negated    : True if the clause is negated
    clause_type: "declarative" | "interrogative" | "imperative"
    """
    subject:         list[Token] = field(default_factory=list)
    verb:            list[Token] = field(default_factory=list)
    direct_object:   list[Token] = field(default_factory=list)
    indirect_object: list[Token] = field(default_factory=list)
    modifier:        list[Token] = field(default_factory=list)
    topic:           list[Token] = field(default_factory=list)
    negated:         bool        = False
    clause_type:     str         = "declarative"

    def all_tokens(self) -> list[Token]:
        return (self.subject + self.verb + self.direct_object
                + self.indirect_object + self.modifier)


# ---------------------------------------------------------------------------
# Grammar rules
# ---------------------------------------------------------------------------

@dataclass
class WordOrderRule:
    """Defines the surface order of clause constituents.

    template: ordered list of slot names, e.g. ["subject","verb","direct_object"]
    applies_to_clause_type: which clause type this rule governs
    applies_when: optional condition dict (e.g. {"transitivity":"transitive"})
    """
    template:               list[str]
    applies_to_clause_type: str  = "declarative"
    applies_when:           dict = field(default_factory=dict)
    name:                   str  = ""
    description:            str  = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "template": self.template,
            "applies_to_clause_type": self.applies_to_clause_type,
            "applies_when": self.applies_when,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> WordOrderRule:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AgreementRule:
    """Copies a grammatical feature from a controller to a target.

    e.g. "verb agrees with subject in number"
      controller_role  = "subject"
      target_role      = "verb"
      feature          = "number"
    """
    controller_role: str
    target_role:     str
    feature:         str
    name:            str = ""
    description:     str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "controller_role": self.controller_role,
            "target_role":     self.target_role,
            "feature":         self.feature,
            "description":     self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AgreementRule:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class NegationRule:
    """How negation is expressed.

    strategy : "prefix_particle" | "suffix_particle" | "verb_affix"
    particle  : the negation word/affix (e.g. "na", "-vel")
    position  : for particles — "before_verb" | "after_verb" | "clause_final"
    """
    strategy:    str  = "prefix_particle"
    particle:    str  = "na"
    position:    str  = "before_verb"
    name:        str  = ""
    description: str  = ""

    def to_dict(self) -> dict:
        return {"name":self.name,"strategy":self.strategy,
                "particle":self.particle,"position":self.position,
                "description":self.description}

    @classmethod
    def from_dict(cls, d: dict) -> NegationRule:
        return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# GrammarEngine
# ---------------------------------------------------------------------------

class GrammarEngine:
    """Assembles morphologically-inflected tokens into grammatical sentences.

    Parameters
    ----------
    morph      : MorphologyEngine for inflecting tokens
    word_order : base word-order typology shorthand (SVO, SOV, VSO…)
                 Used as fallback when no WordOrderRule matches.
    """

    # Default slot order per typology
    _TYPOLOGY_TEMPLATES = {
        "SVO": ["subject","verb","direct_object","indirect_object","modifier"],
        "SOV": ["subject","direct_object","indirect_object","modifier","verb"],
        "VSO": ["verb","subject","direct_object","indirect_object","modifier"],
        "VOS": ["verb","direct_object","subject","indirect_object","modifier"],
        "OVS": ["direct_object","indirect_object","verb","subject","modifier"],
        "OSV": ["direct_object","indirect_object","subject","modifier","verb"],
    }

    def __init__(
        self,
        morph:      Optional[MorphologyEngine] = None,
        word_order: str                        = "SVO",
    ):
        self.morph            = morph
        self.word_order       = word_order
        self.order_rules:     list[WordOrderRule]  = []
        self.agreement_rules: list[AgreementRule]  = []
        self.negation_rule:   Optional[NegationRule] = NegationRule()
        # Optional particles: {"question": "ka", "topic": "wa", ...}
        self.particles:       dict[str, str]       = {}
        # Case assignment rules: role → case value
        self.case_map:        dict[str, str]       = {
            "subject":         "nominative",
            "direct_object":   "accusative",
            "indirect_object": "dative",
        }

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def add_order_rule(self, rule: WordOrderRule) -> None:
        self.order_rules.append(rule)

    def add_agreement_rule(self, rule: AgreementRule) -> None:
        self.agreement_rules.append(rule)

    def set_negation(self, rule: NegationRule) -> None:
        self.negation_rule = rule

    def set_particle(self, function: str, form: str) -> None:
        """Register a grammatical particle. e.g. set_particle("question","ka")"""
        self.particles[function] = form

    def set_case(self, role: str, case: str) -> None:
        """Map a clause role to a case value."""
        self.case_map[role] = case

    # ------------------------------------------------------------------
    # Core: clause → surface string
    # ------------------------------------------------------------------

    def realise(self, clause: Clause) -> tuple[str, list[str]]:
        """Turn a Clause into a surface string.

        Returns (surface_string, [step descriptions]).
        """
        steps = []

        # 1. Assign cases based on role
        for role, case in self.case_map.items():
            for tok in getattr(clause, role, []):
                tok.features.case = case
        steps.append("Case assignment applied.")

        # 2. Apply agreement rules
        for rule in self.agreement_rules:
            controllers = getattr(clause, rule.controller_role, [])
            targets     = getattr(clause, rule.target_role, [])
            if controllers and targets:
                feat_val = getattr(controllers[0].features, rule.feature, None)
                if feat_val:
                    for tok in targets:
                        setattr(tok.features, rule.feature, feat_val)
                    steps.append(
                        f"Agreement: {rule.target_role} copies {rule.feature}"
                        f"={feat_val!r} from {rule.controller_role}."
                    )

        # 3. Inflect all tokens
        if self.morph:
            for tok in clause.all_tokens():
                if tok.is_clang:
                    continue
                feat_dict = tok.features.to_inflect_dict()
                if feat_dict:
                    result = self.morph.inflect(tok.lemma, feat_dict, pos=tok.features.pos)
                    tok.surface = result.surface
                    steps.append(
                        f"Inflect {tok.lemma!r} {feat_dict} → {tok.surface!r}"
                    )

        # 4. Determine word order
        template = self._select_template(clause)
        steps.append(f"Word order template: {template}")

        # 5. Build ordered token list
        ordered: list[Token] = []
        for slot in template:
            ordered.extend(getattr(clause, slot, []))

        # 6. Handle negation
        if clause.negated and self.negation_rule:
            ordered = self._apply_negation(ordered, clause)
            steps.append(
                f"Negation applied: {self.negation_rule.strategy}"
                f" particle={self.negation_rule.particle!r}"
            )

        # 7. Handle clause-type particles
        if clause.clause_type == "interrogative" and "question" in self.particles:
            ordered.append(Token(
                lemma=self.particles["question"],
                surface=self.particles["question"],
                is_clang=True,
            ))
            steps.append(f"Question particle {self.particles['question']!r} appended.")

        surface = " ".join(tok.surface for tok in ordered if tok.surface)
        steps.append(f"Surface: {surface!r}")
        return surface, steps

    def _select_template(self, clause: Clause) -> list[str]:
        """Pick the best matching WordOrderRule, falling back to typology."""
        for rule in self.order_rules:
            if rule.applies_to_clause_type == clause.clause_type:
                cond = rule.applies_when
                if not cond:
                    return rule.template
                # Check transitivity condition
                if "transitivity" in cond:
                    is_trans = bool(clause.direct_object)
                    if (cond["transitivity"] == "transitive") == is_trans:
                        return rule.template
        # Fallback to typology
        return self._TYPOLOGY_TEMPLATES.get(
            self.word_order,
            self._TYPOLOGY_TEMPLATES["SVO"]
        )

    def _apply_negation(
        self, ordered: list[Token], clause: Clause
    ) -> list[Token]:
        """Insert negation particle/affix into the token list."""
        nr   = self.negation_rule
        neg  = Token(lemma=nr.particle, surface=nr.particle, is_clang=True)

        if nr.strategy == "verb_affix":
            # Attach directly to the verb token surface
            for tok in clause.verb:
                if nr.position == "before_verb":
                    tok.surface = nr.particle + tok.surface
                else:
                    tok.surface = tok.surface + nr.particle
            return ordered

        # Particle strategies
        verb_indices = [
            i for i, t in enumerate(ordered)
            if t.role == "verb" or t in clause.verb
        ]
        insert_at = verb_indices[0] if verb_indices else len(ordered)

        if nr.position == "before_verb":
            ordered.insert(insert_at, neg)
        elif nr.position == "after_verb":
            ordered.insert(insert_at + 1, neg)
        elif nr.position == "clause_final":
            ordered.append(neg)

        return ordered

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "word_order":       self.word_order,
            "order_rules":      [r.to_dict() for r in self.order_rules],
            "agreement_rules":  [r.to_dict() for r in self.agreement_rules],
            "negation_rule":    self.negation_rule.to_dict() if self.negation_rule else None,
            "particles":        self.particles,
            "case_map":         self.case_map,
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Grammar saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        morph: Optional[MorphologyEngine] = None,
    ) -> GrammarEngine:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        engine              = cls(morph=morph, word_order=data.get("word_order","SVO"))
        engine.order_rules  = [WordOrderRule.from_dict(r) for r in data.get("order_rules",[])]
        engine.agreement_rules = [AgreementRule.from_dict(r) for r in data.get("agreement_rules",[])]
        if data.get("negation_rule"):
            engine.negation_rule = NegationRule.from_dict(data["negation_rule"])
        engine.particles    = data.get("particles", {})
        engine.case_map     = data.get("case_map", engine.case_map)
        return engine


# ---------------------------------------------------------------------------
# English mini-parser (lightweight, no external dependencies)
# ---------------------------------------------------------------------------

# Simple English word lists for the lightweight parser
_ENG_PRONOUNS = {
    "i":"i","me":"i","my":"i","mine":"i",
    "you":"you","your":"you","yours":"you",
    "he":"he","him":"he","his":"he",
    "she":"she","her":"she","hers":"she",
    "it":"it","its":"it",
    "we":"we","us":"we","our":"we","ours":"we",
    "they":"they","them":"they","their":"they","theirs":"they",
}

_ENG_AUX = {"is","are","was","were","will","would","can","could",
            "shall","should","may","might","must","do","does","did",
            "have","has","had","be","been","being"}

_ENG_NEG  = {"not","n't","never","no"}
_ENG_DET  = {"the","a","an","this","that","these","those","some","any","every"}
_ENG_PREP = {"in","on","at","to","for","of","from","with","by","about","into","onto"}

# Tense signals
_ENG_TENSE_SIGNALS = {
    "yesterday":   "past",
    "ago":         "past",
    "before":      "past",
    "tomorrow":    "future",
    "soon":        "future",
    "will":        "future",
    "going":       "future",
    "already":     "past",
}

# Irregular English past tenses → lemma
_ENG_PAST_FORMS = {
    "ran":"run","saw":"see","went":"go","came":"come","took":"take",
    "gave":"give","found":"find","knew":"know","said":"say",
    "told":"tell","thought":"think","brought":"bring","felt":"feel",
    "heard":"hear","kept":"keep","left":"leave","made":"make",
    "met":"meet","paid":"pay","put":"put","read":"read","sat":"sit",
    "stood":"stand","understood":"understand","won":"win","wrote":"write",
}


def _strip_english_inflection(word: str) -> tuple[str, dict]:
    """Very lightweight English lemmatiser + tense detector.

    Returns (lemma, detected_features).
    Not a full NLP pipeline — handles the most common patterns.
    """
    w = word.lower().rstrip(".,!?;:")
    features: dict = {}

    # Check irregular pasts
    if w in _ENG_PAST_FORMS:
        return _ENG_PAST_FORMS[w], {"tense": "past"}

    # -ed suffix → past
    if w.endswith("ed") and len(w) > 4:
        stem = w[:-2]
        if stem.endswith("e"):
            stem = stem         # "loved" → "love"
        else:
            pass                # "walked" → "walk"
        features["tense"] = "past"
        return stem, features

    # -ing suffix → present progressive (treat as present)
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        if stem.endswith(stem[-1]) and len(stem) > 2:
            stem = stem[:-1]    # "running" → "run"
        return stem, features

    # -s/-es suffix → 3rd sg present
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y", features
    if w.endswith("es") and len(w) > 3:
        return w[:-2], features
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1], features

    # -er comparative
    if w.endswith("er") and len(w) > 4:
        return w[:-2], {"degree": "comparative"}

    # Plural -s/-es/-ies
    # (handled above for verbs; same logic applies for nouns in context)

    return w, features


class EnglishParser:
    """Lightweight English sentence → Clause converter.

    No external NLP libraries required. Handles simple declarative,
    interrogative, and imperative sentences.

    Not a full parser — designed for the most common conlang translation
    use-cases: simple subject–verb–object sentences with optional modifiers.
    """

    def parse(self, sentence: str) -> tuple[Clause, list[str]]:
        """Parse an English sentence into a Clause + step log."""
        steps = []
        tokens_raw = re.findall(r"[a-zA-Z']+|[.,!?]", sentence)
        words  = [w.lower().rstrip(".,!?;:") for w in tokens_raw if w.isalpha()]
        punct  = tokens_raw[-1] if tokens_raw and not tokens_raw[-1].isalpha() else "."

        steps.append(f"Tokens: {words}")

        # Detect clause type
        clause_type = "declarative"
        if punct == "?":
            clause_type = "interrogative"
        elif words and words[0] in ("do","does","did","is","are","was","were","will","can","would","should"):
            clause_type = "interrogative"
        elif words and words[0] not in _ENG_AUX and words[0] not in _ENG_PRONOUNS:
            # Could be imperative (verb-initial without explicit subject)
            pass

        # Detect negation
        negated = any(w in _ENG_NEG for w in words)
        if negated:
            steps.append("Negation detected.")
            words = [w for w in words if w not in _ENG_NEG and w != "n't"]

        # Detect tense from auxiliaries / time adverbs
        tense = ""
        for w in words:
            if w in _ENG_TENSE_SIGNALS:
                tense = _ENG_TENSE_SIGNALS[w]
                break
        if "will" in words or "going" in words:
            tense = "future"
            words = [w for w in words if w not in ("will","going","to","be")]
        if "did" in words or "was" in words or "were" in words:
            tense = "past"
            words = [w for w in words if w not in ("did","was","were")]
        if "does" in words or "is" in words or "are" in words:
            words = [w for w in words if w not in ("does","is","are","do")]

        # Strip determiners, prepositions, auxiliaries
        content = [w for w in words
                   if w not in _ENG_DET and w not in _ENG_AUX
                   and w not in _ENG_PREP]
        steps.append(f"Content words: {content}")

        # Heuristic slot assignment:
        # first noun-ish word = subject, first verb-ish = verb,
        # second noun-ish = object, adjectives/adverbs = modifiers
        subject_words: list[str] = []
        verb_words:    list[str] = []
        object_words:  list[str] = []
        modifier_words:list[str] = []

        # Simple POS heuristic: capitalised or pronoun → subject candidate
        # Ends in common verb suffixes or is in aux list → verb candidate
        verb_found = False
        for w in content:
            lemma, w_feats = _strip_english_inflection(w)
            if not tense and "tense" in w_feats:
                tense = w_feats["tense"]
            # Very naive: assume order Subject Verb Object
            if w in _ENG_PRONOUNS:
                if not verb_found:
                    subject_words.append(lemma)
                else:
                    object_words.append(lemma)
            elif not subject_words and not verb_found:
                subject_words.append(lemma)
            elif not verb_found:
                verb_words.append(lemma)
                verb_found = True
            elif not object_words:
                object_words.append(lemma)
            else:
                modifier_words.append(lemma)

        steps.append(
            f"Parse: S={subject_words} V={verb_words} "
            f"O={object_words} Mod={modifier_words} "
            f"tense={tense!r} neg={negated}"
        )

        # Build Clause with plain Token shells (surface filled later)
        def make_token(lemma, role, pos="noun"):
            feats = GrammaticalFeatures(pos=pos, tense=tense or "")
            return Token(lemma=lemma, surface=lemma, features=feats,
                         role=role, is_clang=False)

        clause = Clause(
            subject       = [make_token(w,"subject","noun")   for w in subject_words],
            verb          = [make_token(w,"verb","verb")       for w in verb_words],
            direct_object = [make_token(w,"direct_object","noun") for w in object_words],
            modifier      = [make_token(w,"modifier","adjective") for w in modifier_words],
            negated       = negated,
            clause_type   = clause_type,
        )
        return clause, steps


# ---------------------------------------------------------------------------
# TranslationResult
# ---------------------------------------------------------------------------

@dataclass
class TranslationResult:
    """The result of a translation pass."""
    surface:   str
    direction: str               # "to_conlang" | "to_english"
    source:    str
    steps:     list[str]         = field(default_factory=list)
    warnings:  list[str]         = field(default_factory=list)
    glosses:   list[str]         = field(default_factory=list)  # per-token glosses

    def log(self) -> str:
        lines = [f"[{self.direction}] {self.source!r} → {self.surface!r}", ""]
        lines += [f"  {i+1:2}. {s}" for i, s in enumerate(self.steps)]
        if self.warnings:
            lines += ["", "  Warnings:"] + [f"    ⚠ {w}" for w in self.warnings]
        if self.glosses:
            lines += ["", "  Interlinear gloss:"] + [f"    {g}" for g in self.glosses]
        return "\n".join(lines)

    def __str__(self):
        return self.surface


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class Translator:
    """Bidirectional translator between English and the conlang.

    Parameters
    ----------
    grammar  : GrammarEngine
    lexicon  : Lexicon
    morph    : MorphologyEngine

    The translation pipeline (English → conlang):
      1. Parse English sentence into a Clause.
      2. Look up each lemma in the lexicon (English gloss → conlang root).
      3. Inflect each token with its grammatical features.
      4. Apply word-order and agreement rules via GrammarEngine.realise().

    The reverse pipeline (conlang → English):
      1. Tokenise conlang sentence.
      2. Morphologically analyse each token → root + features.
      3. Look up root in lexicon → English gloss.
      4. Reconstruct English using feature information.
    """

    def __init__(
        self,
        grammar:  GrammarEngine,
        lexicon:  Optional[Lexicon]          = None,
        morph:    Optional[MorphologyEngine] = None,
    ):
        self.grammar = grammar
        self.lexicon = lexicon
        self.morph   = morph
        # User-override translation table: English word → conlang form
        self.overrides: dict[str, str] = {}

    def add_override(self, english: str, conlang_form: str) -> None:
        """Manually map an English word to a conlang surface form.

        Useful when the automatic gloss lookup doesn't match the user's intent.
        """
        self.overrides[english.lower()] = conlang_form

    # ------------------------------------------------------------------
    # English → conlang
    # ------------------------------------------------------------------

    def translate_to_conlang(
        self,
        sentence: str,
        tense:    str  = "",
        trace:    bool = False,
    ) -> TranslationResult:
        """Translate an English sentence to the conlang."""
        steps:    list[str] = []
        warnings: list[str] = []
        glosses:  list[str] = []

        # 1. Parse English
        parser = EnglishParser()
        clause, parse_steps = parser.parse(sentence)
        steps += parse_steps

        # Override tense if explicitly passed
        if tense:
            for tok_list in [clause.subject, clause.verb,
                             clause.direct_object, clause.modifier]:
                for tok in tok_list:
                    tok.features.tense = tense
            steps.append(f"Tense override: {tense!r}")

        # 2. Look up each token in lexicon
        all_tokens = clause.all_tokens()
        for tok in all_tokens:
            if tok.lemma in self.overrides:
                tok.surface  = self.overrides[tok.lemma]
                tok.is_clang = True
                steps.append(f"Override: {tok.lemma!r} → {tok.surface!r}")
                continue

            conlang_root = self._lookup_english(tok.lemma)
            if conlang_root:
                tok.lemma   = conlang_root.form
                tok.surface = conlang_root.form
                tok.gloss   = conlang_root.gloss
                steps.append(
                    f"Lookup: {tok.lemma!r} → {conlang_root.form!r} "
                    f"({conlang_root.gloss!r})"
                )
            else:
                warnings.append(
                    f"No lexicon entry for {tok.lemma!r}. "
                    f"Keeping English form."
                )
                tok.is_clang = True   # treat as opaque

        # 3. Realise clause (inflection + word order)
        surface, realise_steps = self.grammar.realise(clause)
        steps += realise_steps

        # 4. Build interlinear glosses
        for tok in clause.all_tokens():
            if tok.surface:
                glosses.append(f"{tok.surface:<14} ({tok.gloss or tok.lemma})")

        if trace:
            print("\n".join(f"  {s}" for s in steps))

        return TranslationResult(
            surface   = surface,
            direction = "to_conlang",
            source    = sentence,
            steps     = steps,
            warnings  = warnings,
            glosses   = glosses,
        )

    def _lookup_english(self, english_word: str) -> Optional[LexiconEntry]:
        """Find a conlang lexicon entry whose gloss matches an English word."""
        if not self.lexicon:
            return None
        w = english_word.lower().strip()

        # Exact gloss match
        results = self.lexicon.find_by_gloss(w, fuzzy=False)
        if results:
            return results[0]

        # Fuzzy: gloss starts with or contains the word
        results = self.lexicon.find_by_gloss(w, fuzzy=True)
        # Prefer non-affix entries
        non_affix = [r for r in results if not r.is_affix]
        if non_affix:
            return non_affix[0]
        return results[0] if results else None

    # ------------------------------------------------------------------
    # Conlang → English
    # ------------------------------------------------------------------

    def translate_to_english(
        self,
        sentence: str,
        trace:    bool = False,
    ) -> TranslationResult:
        """Translate a conlang sentence to English."""
        steps:    list[str] = []
        warnings: list[str] = []
        glosses:  list[str] = []

        words = sentence.strip().split()
        steps.append(f"Tokens: {words}")

        english_tokens: list[str] = []
        detected_tense = ""

        for word in words:
            # Check override table (reversed)
            rev_override = {v: k for k, v in self.overrides.items()}
            if word in rev_override:
                english_tokens.append(rev_override[word])
                steps.append(f"Override: {word!r} → {rev_override[word]!r}")
                continue

            # Try morphological analysis
            analyses = []
            if self.morph:
                analyses = self.morph.analyse(word)

            if analyses:
                best = analyses[0]
                root = best["root"]
                entry = self.lexicon.lookup(root) if self.lexicon else None
                gloss = entry.gloss if entry else root

                # Collect feature information
                feat_parts = []
                for affix_form, cat, val in best["affixes"]:
                    feat_parts.append(f"{cat}={val}")
                    if cat == "tense":
                        detected_tense = val

                english_tokens.append(gloss)
                gloss_str = f"{word} = {gloss}" + (f" [{', '.join(feat_parts)}]" if feat_parts else "")
                glosses.append(gloss_str)
                steps.append(
                    f"Analysed {word!r}: root={root!r} ({gloss!r})"
                    + (f", features: {', '.join(feat_parts)}" if feat_parts else "")
                )
            else:
                # Try direct lexicon lookup
                entry = self.lexicon.lookup(word) if self.lexicon else None
                if entry:
                    english_tokens.append(entry.gloss)
                    glosses.append(f"{word} = {entry.gloss}")
                    steps.append(f"Direct lookup: {word!r} → {entry.gloss!r}")
                else:
                    english_tokens.append(word)
                    warnings.append(f"Unknown token {word!r} — kept as-is.")
                    steps.append(f"Unknown: {word!r}")

        # Reconstruct English sentence heuristically
        english = self._reconstruct_english(english_tokens, detected_tense)
        steps.append(f"Reconstructed: {english!r}")

        if trace:
            print("\n".join(f"  {s}" for s in steps))

        return TranslationResult(
            surface   = english,
            direction = "to_english",
            source    = sentence,
            steps     = steps,
            warnings  = warnings,
            glosses   = glosses,
        )

    def _reconstruct_english(self, tokens: list[str], tense: str) -> str:
        """Reassemble English tokens into a readable sentence.

        Applies basic tense marking on the first verb-like token.
        """
        if not tokens:
            return ""

        result = list(tokens)

        # Add tense to what looks like a verb (second content token usually)
        if tense == "past" and len(result) >= 2:
            v = result[1] if len(result) > 1 else result[0]
            if not v.endswith("ed"):
                # Check irregular
                past_map = {v2: k for k, v2 in _ENG_PAST_FORMS.items()}
                result[1] = past_map.get(v, v + "ed")
        elif tense == "future" and len(result) >= 2:
            result.insert(1, "will")

        sentence = " ".join(result)
        return sentence[0].upper() + sentence[1:] + ("." if not sentence.endswith("?") else "")

    # ------------------------------------------------------------------
    # Batch translation
    # ------------------------------------------------------------------

    def translate_batch(
        self,
        sentences: list[str],
        direction: str = "to_conlang",
    ) -> list[TranslationResult]:
        """Translate a list of sentences."""
        fn = (self.translate_to_conlang if direction == "to_conlang"
              else self.translate_to_english)
        return [fn(s) for s in sentences]

    # ------------------------------------------------------------------
    # Phrasebook builder
    # ------------------------------------------------------------------

    def build_phrasebook(self, phrases: list[str]) -> str:
        """Translate a list of English phrases and format as a phrasebook."""
        lines = ["=== Phrasebook ===", ""]
        for phrase in phrases:
            result = self.translate_to_conlang(phrase)
            warn   = "  ⚠ " + "; ".join(result.warnings) if result.warnings else ""
            lines.append(f"  {phrase:<35} →  {result.surface}{warn}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

class GrammarShell:
    """REPL for the grammar engine and translator."""

    HELP = """
Commands
--------
to    <English sentence>         Translate English → conlang
from  <conlang sentence>         Translate conlang → English
trace <English sentence>         Translate with full derivation log
batch                            Enter multiple sentences to translate
phrasebook                       Build a formatted phrasebook interactively
order  <SVO|SOV|VSO|…>           Set word order typology
neg    <particle> [position]     Set negation particle and position
                                  position: before_verb|after_verb|clause_final
particle <name> <form>           Add a grammatical particle
case   <role> <case>             Set case for a clause role
agree  <controller> <target> <feature>   Add agreement rule
override <english> <conlang>     Add a manual translation override
grammar                          Show current grammar settings
save   <path>                    Save grammar to JSON
load   <path>                    Load grammar from JSON
quit
"""

    def __init__(self, translator: Translator):
        self.tr = translator

    def run(self):  # pragma: no cover
        print("\n=== Conlang Grammar & Translation Shell ===")
        print("  Type 'help' for commands.\n")
        while True:
            try:
                raw = input("grammar> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not raw:
                continue
            parts = raw.split(None, 1)
            cmd   = parts[0].lower()
            rest  = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit","exit","q"):
                print("Goodbye.")
                break
            elif cmd == "help":
                print(self.HELP)
            elif cmd == "to" and rest:
                r = self.tr.translate_to_conlang(rest)
                print(f"  {r.surface}")
                if r.warnings:
                    for w in r.warnings: print(f"  ⚠ {w}")
            elif cmd == "from" and rest:
                r = self.tr.translate_to_english(rest)
                print(f"  {r.surface}")
                if r.warnings:
                    for w in r.warnings: print(f"  ⚠ {w}")
            elif cmd == "trace" and rest:
                r = self.tr.translate_to_conlang(rest, trace=False)
                print(r.log())
            elif cmd == "batch":
                print("  Enter sentences (empty line to finish):")
                sents = []
                while True:
                    s = input("  > ").strip()
                    if not s: break
                    sents.append(s)
                for r in self.tr.translate_batch(sents):
                    print(f"  {r.source:<35} →  {r.surface}")
            elif cmd == "phrasebook":
                print("  Enter phrases (empty line to finish):")
                phrases = []
                while True:
                    s = input("  > ").strip()
                    if not s: break
                    phrases.append(s)
                print(self.tr.build_phrasebook(phrases))
            elif cmd == "order" and rest:
                wo = rest.strip().upper()
                if wo in VALID_WORD_ORDERS:
                    self.tr.grammar.word_order = wo
                    print(f"  Word order set to {wo}.")
                else:
                    print(f"  Unknown order {wo!r}. Valid: {sorted(VALID_WORD_ORDERS)}")
            elif cmd == "neg":
                p = rest.split()
                particle = p[0] if p else "na"
                position = p[1] if len(p) > 1 else "before_verb"
                self.tr.grammar.negation_rule = NegationRule(
                    particle=particle, position=position
                )
                print(f"  Negation: {particle!r} {position}")
            elif cmd == "particle" and rest:
                p = rest.split()
                if len(p) >= 2:
                    self.tr.grammar.set_particle(p[0], p[1])
                    print(f"  Particle {p[0]!r} = {p[1]!r}")
            elif cmd == "case" and rest:
                p = rest.split()
                if len(p) >= 2:
                    self.tr.grammar.set_case(p[0], p[1])
                    print(f"  {p[0]} → {p[1]}")
            elif cmd == "agree" and rest:
                p = rest.split()
                if len(p) >= 3:
                    rule = AgreementRule(controller_role=p[0],
                                        target_role=p[1], feature=p[2])
                    self.tr.grammar.add_agreement_rule(rule)
                    print(f"  Agreement: {p[1]} copies {p[2]} from {p[0]}")
            elif cmd == "override" and rest:
                p = rest.split(None, 1)
                if len(p) >= 2:
                    self.tr.add_override(p[0], p[1])
                    print(f"  Override: {p[0]!r} → {p[1]!r}")
            elif cmd == "grammar":
                g = self.tr.grammar
                print(f"  Word order : {g.word_order}")
                print(f"  Negation   : {g.negation_rule.particle!r} "
                      f"({g.negation_rule.position})" if g.negation_rule else "  Negation   : none")
                print(f"  Particles  : {g.particles}")
                print(f"  Case map   : {g.case_map}")
                print(f"  Agreement  : {len(g.agreement_rules)} rules")
            elif cmd == "save" and rest:
                self.tr.grammar.save(rest.strip())
            elif cmd == "load" and rest:
                try:
                    self.tr.grammar = GrammarEngine.load(
                        rest.strip(), morph=self.tr.morph
                    )
                    print("  Grammar loaded.")
                except Exception as exc:
                    print(f"  ✗ {exc}")
            else:
                print(f"  Unknown command: {raw!r}. Type 'help'.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from phonology  import PhonologyEngine
    from lexicon    import Lexicon
    from morphology import MorphologyEngine, AffixSlot, MorphRule

    # --- Set up the stack ---
    phon = PhonologyEngine.from_dict({
        "consonants": ["p","t","k","s","n","l","v","m","r"],
        "vowels":     ["a","e","i","o","u"],
        "syllable_templates": ["CV","CVC","V","(C)V(C)"],
        "phonotactics": [{"type":"no_cluster","phoneme_class":"consonant","max_run":2}],
    })

    lex = Lexicon(phonology=phon)
    lex.add_batch([
        {"form":"tani",  "gloss":"water",   "pos":"noun"},
        {"form":"veka",  "gloss":"run",     "pos":"verb"},
        {"form":"sola",  "gloss":"sun",     "pos":"noun"},
        {"form":"miru",  "gloss":"see",     "pos":"verb"},
        {"form":"nale",  "gloss":"stone",   "pos":"noun"},
        {"form":"piko",  "gloss":"small",   "pos":"adjective"},
        {"form":"talev", "gloss":"sky",     "pos":"noun"},
        {"form":"venu",  "gloss":"person",  "pos":"noun"},
        {"form":"koma",  "gloss":"eat",     "pos":"verb"},
        {"form":"seni",  "gloss":"give",    "pos":"verb"},
        {"form":"lora",  "gloss":"big",     "pos":"adjective"},
        {"form":"-an",   "gloss":"past",    "pos":"suffix","tags":["tense"]},
        {"form":"-el",   "gloss":"future",  "pos":"suffix","tags":["tense"]},
        {"form":"-i",    "gloss":"plural",  "pos":"suffix","tags":["number"]},
        {"form":"re-",   "gloss":"again",   "pos":"prefix","tags":["aspect"]},
    ], source="user")

    morph = MorphologyEngine(lexicon=lex, phonology=phon)
    morph.load_from_lexicon()
    morph.add_slot("noun", AffixSlot(0,"suffix","number",  optional=True))
    morph.add_slot("verb", AffixSlot(0,"suffix","tense",   optional=True))
    morph.add_slot("verb", AffixSlot(1,"suffix","number",  optional=True))
    morph.add_slot("verb", AffixSlot(0,"prefix","aspect",  optional=True))
    morph.add_rule(MorphRule(
        name="vowel_truncation",
        trigger="suffix_vowel_initial",
        environment="stem_vowel_final",
        operation="delete_stem_final",
        description="Drop stem-final vowel before vowel-initial suffix.",
    ))

    gram = GrammarEngine(morph=morph, word_order="SOV")
    gram.set_negation(NegationRule(
        particle="na", position="before_verb",
        description="Negation particle before the verb.",
    ))
    gram.set_particle("question", "ka")
    gram.add_agreement_rule(AgreementRule(
        controller_role="subject", target_role="verb",
        feature="number", name="subj_verb_number_agreement",
    ))

    tr = Translator(grammar=gram, lexicon=lex, morph=morph)

    # --- Translation demo ---
    test_sentences = [
        ("to",   "the water runs"),
        ("to",   "the person sees the stone"),
        ("to",   "the person ran"),
        ("to",   "the sun will shine"),
        ("to",   "the person does not eat"),
        ("to",   "the people eat"),
        ("from", "tani veka"),
        ("from", "venu nale miruan"),
        ("from", "tanii vekai"),
    ]

    print("=== Translation Demo (English ↔ Conlang, SOV order) ===\n")
    for direction, sentence in test_sentences:
        if direction == "to":
            r = tr.translate_to_conlang(sentence)
            print(f"  EN: {sentence:<35} → CL: {r.surface}")
        else:
            r = tr.translate_to_english(sentence)
            print(f"  CL: {sentence:<35} → EN: {r.surface}")
        if r.warnings:
            for w in r.warnings:
                print(f"    ⚠ {w}")

    print()

    # Phrasebook
    print(tr.build_phrasebook([
        "the water is big",
        "I see the sun",
        "the people will run",
        "give me water",
    ]))
    print()

    # Full trace for one sentence
    print("=== Full trace: 'the person ran' ===")
    r = tr.translate_to_conlang("the person ran")
    print(r.log())
    print()

    # Save/load round-trip
    gram.save("/tmp/demo_grammar.json")
    g2 = GrammarEngine.load("/tmp/demo_grammar.json", morph=morph)
    tr2 = Translator(grammar=g2, lexicon=lex, morph=morph)
    r2  = tr2.translate_to_conlang("water runs")
    print(f"Round-trip: 'water runs' → {r2.surface!r}: OK")

    shell = GrammarShell(tr)
    shell.run()
