"""
morphology.py — Conlang Morphology Processor
==============================================
Handles word-internal structure: how roots combine with affixes to produce
inflected or derived surface forms.

Design philosophy
-----------------
Same as the lexicon: the user is the creative authority. The morphology
system lets users:
  • Define affixes they've already decided on (from their lexicon entries).
  • Build named paradigms (declension tables, conjugation tables).
  • Apply morphophonological rules (sound changes at morph boundaries).
  • Inflect any root with a bundle of grammatical features.
  • Inspect the full derivation step-by-step, so they can spot where the
    system diverges from their intuitions and override it.

Components
----------
AffixSlot       : ordered position in a word template (prefix chain / suffix chain)
MorphRule       : a sound-change rule applied at a morph boundary
Paradigm        : a named set of feature-bundles → surface forms (e.g. a noun case table)
MorphologyEngine: top-level manager — inflect, derive, build paradigms, report

Terminology
-----------
  root      : the base lexeme (e.g. "tani")
  affix     : prefix, suffix, or infix from the lexicon
  stem      : root after any stem-change rules, before affixation
  word form : the final surface string after all morphological operations
  gloss     : interlinear morpheme-by-morpheme annotation (Leipzig style)

Usage
-----
    from phonology import PhonologyEngine
    from lexicon   import Lexicon
    from morphology import MorphologyEngine, MorphRule

    phon = PhonologyEngine.load("my_phonology.json")
    lex  = Lexicon.load("my_lexicon.json", phonology=phon)
    morph = MorphologyEngine(lex, phon)

    # Register a morphophonological rule: final vowel of stem deleted
    # before a vowel-initial suffix
    morph.add_rule(MorphRule(
        name        = "vowel_truncation",
        trigger     = "suffix_vowel_initial",
        environment = "stem_vowel_final",
        operation   = "delete_stem_final",
        description = "Drop stem-final vowel before vowel-initial suffix",
    ))

    # Inflect a root
    result = morph.inflect("tani", {"number": "plural"})
    print(result.surface)   # "tanii" or whatever the rule produces
    print(result.gloss)     # "tani-i  water-PL"

    # Build a full paradigm
    paradigm = morph.build_paradigm("noun_basic", "tani",
        feature_sets=[
            {"number": "singular"},
            {"number": "plural"},
        ]
    )
    print(paradigm.table())
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from phonology import PhonologyEngine
except ImportError:
    PhonologyEngine = None  # type: ignore

try:
    from lexicon import Lexicon, LexiconEntry
except ImportError:
    Lexicon = None          # type: ignore
    LexiconEntry = None     # type: ignore


# ---------------------------------------------------------------------------
# MorphRule  — sound changes at morpheme boundaries
# ---------------------------------------------------------------------------

VALID_TRIGGERS = {
    "always",               # apply unconditionally
    "suffix_vowel_initial", # the suffix being attached starts with a vowel
    "suffix_cons_initial",  # the suffix starts with a consonant
    "prefix_vowel_final",   # the prefix ends with a vowel
    "prefix_cons_final",    # the prefix ends with a consonant
    "stem_vowel_final",     # the stem ends with a vowel
    "stem_cons_final",      # the stem ends with a consonant
}

VALID_OPERATIONS = {
    "delete_stem_final",    # remove the last character of the stem
    "delete_affix_initial", # remove the first character of the affix
    "insert",               # insert a string (specified in params["insert"])
    "replace_stem_final",   # replace last char of stem (params["replacement"])
    "replace_affix_initial",# replace first char of affix
    "geminate_boundary",    # double the consonant at the junction
}


@dataclass
class MorphRule:
    """A morphophonological rule applied when two morphemes meet.

    name        : unique identifier
    trigger     : when this rule fires (see VALID_TRIGGERS)
    environment : additional condition that must also hold (or "always")
    operation   : what to do (see VALID_OPERATIONS)
    params      : extra data needed by some operations (e.g. {"insert": "n"})
    description : human-readable explanation
    applies_to  : restrict to specific POS or affix forms; empty = unrestricted
    """
    name: str
    trigger: str                  = "always"
    environment: str              = "always"
    operation: str                = "delete_stem_final"
    params: dict                  = field(default_factory=dict)
    description: str              = ""
    applies_to: list[str]         = field(default_factory=list)

    def __post_init__(self):
        if self.trigger not in VALID_TRIGGERS:
            raise ValueError(f"Unknown trigger {self.trigger!r}. Valid: {sorted(VALID_TRIGGERS)}")
        if self.operation not in VALID_OPERATIONS:
            raise ValueError(f"Unknown operation {self.operation!r}. Valid: {sorted(VALID_OPERATIONS)}")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "trigger": self.trigger,
            "environment": self.environment,
            "operation": self.operation,
            "params": self.params,
            "description": self.description,
            "applies_to": self.applies_to,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MorphRule:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# InflectionResult  — what inflect() returns
# ---------------------------------------------------------------------------

@dataclass
class InflectionResult:
    """The result of a single inflection operation.

    surface     : the final surface form (e.g. "taniin")
    root        : the original root (e.g. "tani")
    features    : the feature bundle that was applied
    steps       : ordered list of (description, intermediate_form) for tracing
    morphemes   : list of (morpheme_string, gloss_label) in order
    gloss       : interlinear gloss string (Leipzig notation)
    warnings    : any issues encountered during inflection
    """
    surface: str
    root: str
    features: dict
    steps: list[tuple[str, str]]    = field(default_factory=list)
    morphemes: list[tuple[str, str]]= field(default_factory=list)
    warnings: list[str]             = field(default_factory=list)

    @property
    def gloss(self) -> str:
        """Build a Leipzig-style interlinear gloss."""
        parts  = "-".join(m for m, _ in self.morphemes)
        labels = "-".join(g for _, g in self.morphemes)
        return f"{parts}\n{labels}"

    def trace(self) -> str:
        """Print the full derivation step-by-step."""
        lines = [f"Root: {self.root!r}   Features: {self.features}"]
        for desc, form in self.steps:
            lines.append(f"  → [{desc}] {form!r}")
        lines.append(f"Surface: {self.surface!r}")
        if self.warnings:
            lines.append("Warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paradigm  — a full inflection table for one root
# ---------------------------------------------------------------------------

@dataclass
class Paradigm:
    """A named inflection table: feature-bundle → InflectionResult."""
    name: str
    root: str
    cells: list[tuple[dict, InflectionResult]] = field(default_factory=list)

    def table(self, show_gloss: bool = False) -> str:
        """Pretty-print the paradigm as a table."""
        lines = [f"=== Paradigm: {self.name}  (root: {self.root!r}) ==="]
        for features, result in self.cells:
            feat_str = "  ".join(f"{k}={v}" for k, v in features.items())
            lines.append(f"  {feat_str:<30} →  {result.surface}")
            if show_gloss:
                for line in result.gloss.split("\n"):
                    lines.append(f"    {line}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "root": self.root,
            "cells": [
                {"features": feats, "surface": res.surface, "gloss": res.gloss}
                for feats, res in self.cells
            ],
        }


# ---------------------------------------------------------------------------
# AffixSlot  — ordered positions in a word template
# ---------------------------------------------------------------------------

@dataclass
class AffixSlot:
    """One position in the ordered affix template for a POS.

    position    : integer order (lower = closer to root for prefixes,
                  lower = closer to root for suffixes — both start at 0)
    kind        : "prefix" | "suffix" | "infix"
    category    : grammatical category this slot carries (e.g. "tense", "number")
    optional    : if True, the slot may be unfilled
    default     : default affix form if no feature is specified (or "")
    """
    position: int
    kind: str                     # "prefix" | "suffix" | "infix"
    category: str                 # e.g. "tense", "number", "case"
    optional: bool                = True
    default: str                  = ""

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "kind": self.kind,
            "category": self.category,
            "optional": self.optional,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AffixSlot:
        return cls(**d)


# ---------------------------------------------------------------------------
# MorphologyEngine
# ---------------------------------------------------------------------------

class MorphologyEngine:
    """Top-level morphology manager.

    Parameters
    ----------
    lexicon   : Lexicon instance with affix entries already populated.
    phonology : PhonologyEngine for sound-change context (optional).
    """

    def __init__(
        self,
        lexicon: Optional[Lexicon]         = None,
        phonology: Optional[PhonologyEngine] = None,
    ):
        self.lexicon   = lexicon
        self.phonology = phonology
        self.rules:    list[MorphRule]  = []
        self.slots:    dict[str, list[AffixSlot]] = {}  # pos → [AffixSlot]
        # Feature → affix form mapping: category → {value: form}
        # e.g. {"tense": {"past": "-an", "future": "-el"}}
        self.feature_map: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_rule(self, rule: MorphRule) -> None:
        """Register a morphophonological rule."""
        self.rules.append(rule)

    def add_slot(self, pos: str, slot: AffixSlot) -> None:
        """Register an affix slot for a POS."""
        self.slots.setdefault(pos, []).append(slot)
        self.slots[pos].sort(key=lambda s: s.position)

    def map_feature(self, category: str, value: str, affix_form: str) -> None:
        """Map a grammatical feature value to an affix form.

        e.g. map_feature("tense", "past", "-an")
             map_feature("number", "plural", "-i")
        """
        self.feature_map.setdefault(category, {})[value] = affix_form

    def load_from_lexicon(self) -> int:
        """Scan the attached lexicon and auto-register affix entries.

        Any lexicon entry with pos in {prefix, suffix, infix} that has a
        non-empty `tags` list will be mapped as:
            first tag → category, gloss label → value

        This lets users build their feature map simply by adding well-tagged
        affix entries to the lexicon. Returns number of affixes registered.
        """
        if not self.lexicon:
            return 0
        count = 0
        for entry in self.lexicon.entries:
            if entry.pos not in ("prefix", "suffix", "infix"):
                continue
            if not entry.tags:
                continue
            category = entry.tags[0]
            # Use the gloss as the value key, normalised to lowercase
            value = entry.gloss.strip().lower()
            self.map_feature(category, value, entry.form)
            count += 1
        return count

    # ------------------------------------------------------------------
    # Core inflection
    # ------------------------------------------------------------------

    def inflect(
        self,
        root: str,
        features: dict,
        pos: str        = "noun",
        trace: bool     = False,
    ) -> InflectionResult:
        """Inflect a root word with a bundle of grammatical features.

        Parameters
        ----------
        root     : the base lexeme (without affixes)
        features : dict mapping category → value, e.g. {"tense": "past", "number": "plural"}
        pos      : part of speech (used to select slot ordering)
        trace    : if True, print derivation steps to stdout

        Returns an InflectionResult with .surface, .gloss, and .steps.
        """
        steps:     list[tuple[str, str]] = [("root", root)]
        morphemes: list[tuple[str, str]] = []
        warnings:  list[str]             = []

        # If the root is flagged as irregular, note it at the top of the trace
        if self.lexicon:
            root_entry = self.lexicon.lookup(root)
            if root_entry and root_entry.warnings:
                reason = root_entry.irregularity_reason or "no reason recorded"
                steps.append(("irregular root", f"{root!r} — {reason}"))
                warnings.append(f"{root!r} is phonologically irregular: {reason}")

        # Resolve features → affix forms
        prefixes: list[tuple[str, str, str]] = []   # (form, category, value)
        suffixes: list[tuple[str, str, str]] = []

        # Respect slot ordering if defined for this POS
        ordered_cats = self._ordered_categories(pos)
        # Add any features not in the slot order at the end
        all_cats = ordered_cats + [c for c in features if c not in ordered_cats]

        for category in all_cats:
            value = features.get(category)
            if value is None:
                continue
            affix_form = self.feature_map.get(category, {}).get(value)
            if affix_form is None:
                warnings.append(
                    f"No affix found for {category}={value!r}. "
                    f"Known values: {list(self.feature_map.get(category, {}).keys())}"
                )
                continue
            if affix_form.startswith("-"):
                suffixes.append((affix_form.lstrip("-"), category, value))
            elif affix_form.endswith("-"):
                prefixes.append((affix_form.rstrip("-"), category, value))
            else:
                # Treat bare forms as suffixes
                suffixes.append((affix_form, category, value))

        # Sort by slot position if available
        prefixes = self._sort_by_slot(prefixes, pos, "prefix")
        suffixes = self._sort_by_slot(suffixes, pos, "suffix")

        # Build the word: [prefixes][stem][suffixes]
        stem = root

        # Prefix chain
        prefix_str = ""
        for form, category, value in prefixes:
            stem_before = stem
            form, stem = self._apply_rules(
                stem=stem, affix=form, direction="prefix"
            )
            prefix_str += form
            steps.append((f"prefix:{category}={value}", prefix_str + stem))
            morphemes.append((form, self._gloss_label(category, value)))

        # Morpheme entry for the root
        root_gloss = ""
        if self.lexicon:
            entry = self.lexicon.lookup(root)
            root_gloss = entry.gloss if entry else root
        morphemes.append((stem, root_gloss))

        # Suffix chain
        for form, category, value in suffixes:
            stem_before = stem
            affix, stem_after = self._apply_rules(
                stem=stem, affix=form, direction="suffix"
            )
            stem = stem + affix
            steps.append((f"suffix:{category}={value}", prefix_str + stem))
            morphemes.append((affix, self._gloss_label(category, value)))

        surface = prefix_str + stem
        steps.append(("surface", surface))

        result = InflectionResult(
            surface   = surface,
            root      = root,
            features  = features,
            steps     = steps,
            morphemes = morphemes,
            warnings  = warnings,
        )

        if trace:
            print(result.trace())

        return result

    def _ordered_categories(self, pos: str) -> list[str]:
        """Return categories in slot order for a POS."""
        slot_list = self.slots.get(pos, [])
        return [s.category for s in sorted(slot_list, key=lambda s: s.position)]

    def _sort_by_slot(
        self,
        affixes: list[tuple[str, str, str]],
        pos: str,
        kind: str,
    ) -> list[tuple[str, str, str]]:
        """Sort affixes by their AffixSlot position."""
        slot_order = {
            s.category: s.position
            for s in self.slots.get(pos, [])
            if s.kind == kind
        }
        return sorted(affixes, key=lambda a: slot_order.get(a[1], 999))

    def _apply_rules(
        self, stem: str, affix: str, direction: str
    ) -> tuple[str, str]:
        """Apply all matching MorphRules at the stem–affix boundary.

        Returns (possibly_modified_affix, possibly_modified_stem).
        For direction="suffix": stem stays, affix may change (or stem end may change).
        For direction="prefix": affix stays, stem start may change.
        """
        vowel_syms = set("aeiouAEIOU")
        if self.phonology:
            vowel_syms = {p.symbol for p in self.phonology.vowels}

        for rule in self.rules:
            # Evaluate trigger
            if direction == "suffix":
                trigger_met = (
                    rule.trigger == "always"
                    or (rule.trigger == "suffix_vowel_initial" and affix and affix[0] in vowel_syms)
                    or (rule.trigger == "suffix_cons_initial"  and affix and affix[0] not in vowel_syms)
                    or (rule.trigger == "stem_vowel_final"     and stem and stem[-1] in vowel_syms)
                    or (rule.trigger == "stem_cons_final"      and stem and stem[-1] not in vowel_syms)
                )
            else:  # prefix
                trigger_met = (
                    rule.trigger == "always"
                    or (rule.trigger == "prefix_vowel_final" and affix and affix[-1] in vowel_syms)
                    or (rule.trigger == "prefix_cons_final"  and affix and affix[-1] not in vowel_syms)
                    or (rule.trigger == "stem_vowel_final"   and stem and stem[0] in vowel_syms)
                    or (rule.trigger == "stem_cons_final"    and stem and stem[0] not in vowel_syms)
                )

            # Evaluate environment
            env_met = (
                rule.environment == "always"
                or (rule.environment == "stem_vowel_final"    and stem and stem[-1] in vowel_syms)
                or (rule.environment == "stem_cons_final"     and stem and stem[-1] not in vowel_syms)
                or (rule.environment == "suffix_vowel_initial" and affix and affix[0] in vowel_syms)
                or (rule.environment == "suffix_cons_initial"  and affix and affix[0] not in vowel_syms)
            )

            if not (trigger_met and env_met):
                continue

            # Apply operation
            op = rule.operation
            if op == "delete_stem_final" and stem:
                stem = stem[:-1]
            elif op == "delete_affix_initial" and affix:
                affix = affix[1:]
            elif op == "insert":
                ins = rule.params.get("insert", "")
                if direction == "suffix":
                    affix = ins + affix
                else:
                    affix = affix + ins
            elif op == "replace_stem_final" and stem:
                repl = rule.params.get("replacement", "")
                stem = stem[:-1] + repl
            elif op == "replace_affix_initial" and affix:
                repl = rule.params.get("replacement", "")
                affix = repl + affix[1:]
            elif op == "geminate_boundary":
                if direction == "suffix" and affix:
                    affix = affix[0] + affix
                elif direction == "prefix" and affix:
                    affix = affix + affix[-1]

        return affix, stem

    @staticmethod
    def _gloss_label(category: str, value: str) -> str:
        """Format a Leipzig-style gloss label (e.g. 'PL', 'PST')."""
        abbreviations = {
            "plural":    "PL",   "singular":  "SG",
            "past":      "PST",  "present":   "PRS",  "future":    "FUT",
            "nominative":"NOM",  "accusative":"ACC",  "genitive":  "GEN",
            "dative":    "DAT",  "locative":  "LOC",  "ablative":  "ABL",
            "masculine": "MASC", "feminine":  "FEM",  "neuter":    "NEUT",
            "1sg": "1SG", "2sg": "2SG", "3sg": "3SG",
            "1pl": "1PL", "2pl": "2PL", "3pl": "3PL",
            "again":     "ITER", "agent":     "AGT",  "passive":   "PASS",
        }
        return abbreviations.get(value.lower(), value.upper())

    # ------------------------------------------------------------------
    # Paradigm builder
    # ------------------------------------------------------------------

    def build_paradigm(
        self,
        name: str,
        root: str,
        feature_sets: list[dict],
        pos: str = "noun",
    ) -> Paradigm:
        """Build a full inflection paradigm for a root.

        Parameters
        ----------
        name         : human-readable name (e.g. "noun_basic")
        root         : the base form
        feature_sets : list of feature bundles, one per cell
        """
        cells = []
        for features in feature_sets:
            result = self.inflect(root, features, pos=pos)
            cells.append((features, result))
        return Paradigm(name=name, root=root, cells=cells)

    # ------------------------------------------------------------------
    # Word derivation (root → new lexeme via derivational morphology)
    # ------------------------------------------------------------------

    def derive(
        self,
        root: str,
        derivation_affix: str,
        new_gloss: str,
        new_pos: str      = "noun",
        add_to_lexicon: bool = True,
    ) -> Optional[LexiconEntry]:
        """Create a derived word using a derivational affix.

        e.g. derive("miru", "-tor", "one who sees → seer", "noun")

        If add_to_lexicon=True and a Lexicon is attached, the derived form
        is added with source="derived" and related=[root].
        """
        if derivation_affix.startswith("-"):
            affix_bare = derivation_affix.lstrip("-")
            affix_str, _ = self._apply_rules(root, affix_bare, "suffix")
            surface = root + affix_str
        elif derivation_affix.endswith("-"):
            affix_bare = derivation_affix.rstrip("-")
            affix_str, _ = self._apply_rules(root, affix_bare, "prefix")
            surface = affix_str + root
        else:
            surface = root + derivation_affix

        print(f"  Derived form: {surface!r}  gloss: {new_gloss!r}  pos: {new_pos}")

        if add_to_lexicon and self.lexicon:
            try:
                entry = self.lexicon.add(
                    form    = surface,
                    gloss   = new_gloss,
                    pos     = new_pos,
                    source  = "derived",
                    related = [root],
                )
                return entry
            except ValueError as exc:
                print(f"  ⚠ Could not add derived form to lexicon: {exc}")
        return None

    # ------------------------------------------------------------------
    # Analysis (morphological parsing in reverse)
    # ------------------------------------------------------------------

    def analyse(self, word: str) -> list[dict]:
        """Attempt to decompose a surface form into root + affixes.

        Returns a list of candidate analyses (there may be ambiguity).
        Each analysis is a dict:
            {"root": str, "affixes": [(form, category, value)], "surface": str}

        This is a brute-force segmentation — try stripping each known affix
        and check if the remainder is in the lexicon.
        """
        if not self.lexicon:
            return []

        candidates = []

        # Collect all known affix bare forms
        all_affixes: list[tuple[str, str, str, str]] = []  # (bare, full, cat, val)
        for cat, val_map in self.feature_map.items():
            for val, full_form in val_map.items():
                bare = full_form.strip("-")
                all_affixes.append((bare, full_form, cat, val))

        # Try stripping suffixes
        for bare, full, cat, val in all_affixes:
            if full.startswith("-") and word.endswith(bare):
                remainder = word[: -len(bare)] if bare else word
                if self.lexicon.lookup(remainder):
                    candidates.append({
                        "root": remainder,
                        "affixes": [(full, cat, val)],
                        "surface": word,
                    })
                # Try two-suffix combos
                for bare2, full2, cat2, val2 in all_affixes:
                    if full2.startswith("-") and remainder.endswith(bare2) and cat2 != cat:
                        inner = remainder[: -len(bare2)] if bare2 else remainder
                        if self.lexicon.lookup(inner):
                            candidates.append({
                                "root": inner,
                                "affixes": [(full2, cat2, val2), (full, cat, val)],
                                "surface": word,
                            })

        # Try stripping prefixes
        for bare, full, cat, val in all_affixes:
            if full.endswith("-") and word.startswith(bare):
                remainder = word[len(bare):]
                if self.lexicon.lookup(remainder):
                    candidates.append({
                        "root": remainder,
                        "affixes": [(full, cat, val)],
                        "surface": word,
                    })

        return candidates

    def analyse_str(self, word: str) -> str:
        """Pretty-print morphological analyses of a word."""
        candidates = self.analyse(word)
        if not candidates:
            return f"  No analysis found for {word!r}."
        lines = [f"  Analyses for {word!r}:"]
        for i, c in enumerate(candidates, 1):
            affixes = "  ".join(
                f"{f} [{cat}={val}]" for f, cat, val in c["affixes"]
            )
            lines.append(f"  {i}. root={c['root']!r}  affixes: {affixes}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Pattern reporting
    # ------------------------------------------------------------------

    def feature_coverage_report(self) -> str:
        """Show which grammatical categories and values are defined."""
        if not self.feature_map:
            return "No features mapped yet. Use map_feature() or load_from_lexicon()."
        lines = ["=== Feature Coverage ==="]
        for cat, val_map in sorted(self.feature_map.items()):
            lines.append(f"  {cat}:")
            for val, form in sorted(val_map.items()):
                lines.append(f"    {val:<20} → {form}")
        return "\n".join(lines)

    def slot_report(self) -> str:
        """Show the slot template for each POS."""
        if not self.slots:
            return "No slots defined."
        lines = ["=== Affix Slot Templates ==="]
        for pos, slot_list in sorted(self.slots.items()):
            lines.append(f"  {pos}:")
            for s in sorted(slot_list, key=lambda x: x.position):
                opt = "(optional)" if s.optional else "(required)"
                lines.append(
                    f"    pos={s.position}  {s.kind:<8}  {s.category:<15} {opt}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "rules":       [r.to_dict() for r in self.rules],
            "slots":       {pos: [s.to_dict() for s in slist]
                            for pos, slist in self.slots.items()},
            "feature_map": self.feature_map,
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Morphology saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        lexicon: Optional[Lexicon]            = None,
        phonology: Optional[PhonologyEngine]  = None,
    ) -> MorphologyEngine:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        engine = cls(lexicon=lexicon, phonology=phonology)
        engine.rules       = [MorphRule.from_dict(r) for r in data.get("rules", [])]
        engine.slots       = {
            pos: [AffixSlot.from_dict(s) for s in slist]
            for pos, slist in data.get("slots", {}).items()
        }
        engine.feature_map = data.get("feature_map", {})
        return engine


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

class MorphologyShell:
    """REPL for the morphology engine."""

    HELP = """
Commands
--------
inflect <root> <cat=val> [cat=val ...]   Inflect a root with features
                                          e.g.  inflect tani number=plural
                                                inflect veka tense=past number=plural
trace   <root> <cat=val> [...]           Same but prints full derivation steps
analyse <word>                           Morphological analysis (parse)
paradigm <root> <pos>                    Print full paradigm for a root+POS
features                                 Show all mapped features
slots                                    Show affix slot templates
derive  <root> <affix> <gloss> [pos]     Create a derived word
map     <category> <value> <affix>       Add/update a feature mapping
                                          e.g.  map tense past -an
addrule                                  Interactively add a morphophonological rule
loadlex                                  Auto-load affixes from lexicon
save    <path>                           Save morphology config to JSON
load    <path>                           Load morphology config from JSON
quit
"""

    def __init__(self, engine: MorphologyEngine):
        self.engine = engine

    def _parse_features(self, tokens: list[str]) -> dict:
        features = {}
        for t in tokens:
            if "=" in t:
                k, v = t.split("=", 1)
                features[k.strip()] = v.strip()
        return features

    def run(self):  # pragma: no cover
        print("\n=== Conlang Morphology Shell ===")
        print("  Type 'help' for commands.\n")
        while True:
            try:
                raw = input("morphology> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not raw:
                continue
            parts = raw.split()
            cmd   = parts[0].lower()

            if cmd in ("quit", "exit", "q"):
                print("Goodbye.")
                break

            elif cmd == "help":
                print(self.HELP)

            elif cmd in ("inflect", "trace") and len(parts) >= 3:
                root     = parts[1]
                features = self._parse_features(parts[2:])
                do_trace = cmd == "trace"
                result   = self.engine.inflect(root, features, trace=do_trace)
                print(f"  {root!r} + {features} → {result.surface!r}")
                if result.warnings:
                    for w in result.warnings:
                        print(f"  ⚠ {w}")
                print(f"  Gloss:\n    {result.gloss}")

            elif cmd == "analyse" and len(parts) >= 2:
                print(self.engine.analyse_str(parts[1]))

            elif cmd == "paradigm" and len(parts) >= 2:
                root = parts[1]
                pos  = parts[2] if len(parts) > 2 else "noun"
                # Build a paradigm from all known feature combos for this POS
                feature_sets = self._feature_sets_for_pos(pos)
                if not feature_sets:
                    print("  No features defined. Use 'features' to check.")
                else:
                    p = self.engine.build_paradigm(f"{root}_{pos}", root, feature_sets, pos)
                    print(p.table(show_gloss=True))

            elif cmd == "features":
                print(self.engine.feature_coverage_report())

            elif cmd == "slots":
                print(self.engine.slot_report())

            elif cmd == "derive" and len(parts) >= 4:
                root  = parts[1]
                affix = parts[2]
                gloss = parts[3]
                pos   = parts[4] if len(parts) > 4 else "noun"
                self.engine.derive(root, affix, gloss, pos)

            elif cmd == "map" and len(parts) >= 4:
                cat, val, affix = parts[1], parts[2], parts[3]
                self.engine.map_feature(cat, val, affix)
                print(f"  Mapped {cat}={val!r} → {affix!r}")

            elif cmd == "addrule":
                self._interactive_add_rule()

            elif cmd == "loadlex":
                n = self.engine.load_from_lexicon()
                print(f"  Loaded {n} affix mappings from lexicon.")

            elif cmd == "save" and len(parts) >= 2:
                self.engine.save(parts[1])

            elif cmd == "load" and len(parts) >= 2:
                try:
                    loaded = MorphologyEngine.load(
                        parts[1],
                        lexicon=self.engine.lexicon,
                        phonology=self.engine.phonology,
                    )
                    self.engine.rules       = loaded.rules
                    self.engine.slots       = loaded.slots
                    self.engine.feature_map = loaded.feature_map
                    print(f"  Loaded morphology from {parts[1]}")
                except Exception as exc:
                    print(f"  ✗ {exc}")

            else:
                print(f"  Unknown command: {raw!r}. Type 'help'.")

    def _feature_sets_for_pos(self, pos: str) -> list[dict]:
        """Build all single-category feature bundles for a POS."""
        feature_sets = []
        for cat, val_map in self.engine.feature_map.items():
            for val in val_map:
                feature_sets.append({cat: val})
        return feature_sets

    def _interactive_add_rule(self):  # pragma: no cover
        print("  Adding a morphophonological rule.")
        print(f"  Triggers: {sorted(VALID_TRIGGERS)}")
        print(f"  Operations: {sorted(VALID_OPERATIONS)}")
        name  = input("  Rule name: ").strip()
        trig  = input("  Trigger: ").strip()
        env   = input("  Environment (or 'always'): ").strip() or "always"
        op    = input("  Operation: ").strip()
        desc  = input("  Description: ").strip()
        params = {}
        if op in ("insert", "replace_stem_final", "replace_affix_initial"):
            val = input("  Parameter value (insert/replacement string): ").strip()
            key = "insert" if op == "insert" else "replacement"
            params[key] = val
        try:
            rule = MorphRule(name=name, trigger=trig, environment=env,
                             operation=op, params=params, description=desc)
            self.engine.add_rule(rule)
            print(f"  ✓ Rule {name!r} added.")
        except ValueError as exc:
            print(f"  ✗ {exc}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # --- Load phonology and lexicon ---
    try:
        from phonology import PhonologyEngine
        from lexicon import Lexicon
    except ImportError:
        print("Could not import phonology/lexicon — running with stubs.")
        PhonologyEngine = None
        Lexicon = None

    phon = PhonologyEngine.from_dict({
        "consonants": ["p","t","k","s","n","l","v","m","r"],
        "vowels":     ["a","e","i","o","u"],
        "syllable_templates": ["CV","CVC","V","(C)V(C)"],
        "phonotactics": [
            {"type": "no_cluster", "phoneme_class": "consonant", "max_run": 2},
        ],
    }) if PhonologyEngine else None

    lex = Lexicon(phonology=phon) if Lexicon else None
    if lex:
        lex.add_batch([
            {"form": "tani",  "gloss": "water",    "pos": "noun"},
            {"form": "veka",  "gloss": "to run",   "pos": "verb"},
            {"form": "sola",  "gloss": "sun",      "pos": "noun"},
            {"form": "miru",  "gloss": "to see",   "pos": "verb"},
            {"form": "nale",  "gloss": "stone",    "pos": "noun"},
            # User-defined affixes — will be auto-loaded by load_from_lexicon()
            {"form": "-an",   "gloss": "past",     "pos": "suffix", "tags": ["tense"]},
            {"form": "-el",   "gloss": "future",   "pos": "suffix", "tags": ["tense"]},
            {"form": "-i",    "gloss": "plural",   "pos": "suffix", "tags": ["number"]},
            {"form": "re-",   "gloss": "again",    "pos": "prefix", "tags": ["aspect"]},
            {"form": "-ok",   "gloss": "agent",    "pos": "suffix", "tags": ["derivation"]},
        ], source="user")

    # --- Build morphology engine ---
    morph = MorphologyEngine(lexicon=lex, phonology=phon)

    # Auto-load affix mappings from lexicon
    n = morph.load_from_lexicon()
    print(f"Auto-loaded {n} affix mappings from lexicon.\n")

    # Define slot ordering for nouns and verbs
    morph.add_slot("noun", AffixSlot(position=0, kind="suffix", category="number",  optional=True))
    morph.add_slot("verb", AffixSlot(position=0, kind="suffix", category="tense",   optional=True))
    morph.add_slot("verb", AffixSlot(position=1, kind="suffix", category="number",  optional=True))
    morph.add_slot("verb", AffixSlot(position=0, kind="prefix", category="aspect",  optional=True))

    # Add a morphophonological rule: drop stem-final vowel before vowel-initial suffix
    morph.add_rule(MorphRule(
        name        = "vowel_truncation",
        trigger     = "suffix_vowel_initial",
        environment = "stem_vowel_final",
        operation   = "delete_stem_final",
        description = "Drop stem-final vowel before a vowel-initial suffix.",
    ))

    print(morph.feature_coverage_report())
    print()
    print(morph.slot_report())
    print()

    # Inflect some words
    tests = [
        ("tani",  {"number": "plural"},           "noun"),
        ("veka",  {"tense": "past"},               "verb"),
        ("veka",  {"tense": "future", "number": "plural"}, "verb"),
        ("miru",  {"aspect": "again", "tense": "past"},    "verb"),
        ("sola",  {"number": "plural"},            "noun"),
    ]
    print("=== Inflection examples ===")
    for root, feats, pos in tests:
        result = morph.inflect(root, feats, pos=pos)
        feat_str = "  ".join(f"{k}={v}" for k, v in feats.items())
        print(f"  {root:<8} [{feat_str}]  →  {result.surface}")
        print(f"    gloss: {result.gloss.splitlines()[1]}")
    print()

    # Full noun paradigm
    noun_paradigm = morph.build_paradigm(
        "noun_number", "tani",
        feature_sets=[{"number": "plural"}, {}],
        pos="noun",
    )
    print(noun_paradigm.table(show_gloss=True))
    print()

    # Verb paradigm
    verb_paradigm = morph.build_paradigm(
        "verb_tense", "veka",
        feature_sets=[
            {"tense": "past"},
            {"tense": "future"},
            {"aspect": "again", "tense": "past"},
        ],
        pos="verb",
    )
    print(verb_paradigm.table(show_gloss=True))
    print()

    # Derivation
    print("=== Derivation ===")
    morph.derive("miru", "-ok", "one who sees → seer", "noun")
    morph.derive("veka", "-ok", "runner", "noun")
    print()

    # Morphological analysis
    print("=== Analysis (reverse parsing) ===")
    for w in ["tanii", "vekaan", "miruel", "sola"]:
        print(morph.analyse_str(w))
    print()

    # Save/load round-trip
    morph.save("/tmp/demo_morphology.json")
    m2 = MorphologyEngine.load("/tmp/demo_morphology.json", lexicon=lex, phonology=phon)
    r2 = m2.inflect("tani", {"number": "plural"})
    assert r2.surface, "Round-trip inflection failed"
    print("Save/load round-trip: OK\n")

    # Launch shell
    shell = MorphologyShell(morph)
    shell.run()
