"""
phonology.py — Conlang Phonology Engine
========================================
Defines and validates the sound system of a constructed language.

Components:
  - Phoneme         : a single sound unit with features
  - SyllableTemplate: a CV-pattern template (e.g. "(C)V(C)")
  - PhonotacticRule : a constraint on sound sequences
  - PhonologyEngine : top-level manager; validates words, generates syllables
  - Romanizer       : maps internal phoneme codes to a user-facing spelling

Usage example:
    from phonology import PhonologyEngine, Phoneme, SyllableTemplate, PhonotacticRule

    eng = PhonologyEngine.from_dict({
        "consonants": ["p","t","k","s","n","l"],
        "vowels":     ["a","e","i","o","u"],
        "syllable_templates": ["CV", "CVC", "V"],
        "phonotactics": [
            {"type": "no_cluster", "phoneme_class": "consonant", "max_run": 2}
        ],
        "romanization": {"k": "k", "s": "s"}   # identity by default
    })

    print(eng.validate_word("tani"))      # ValidationResult(valid=True, ...)
    print(eng.random_word(syllables=2))   # e.g. "nelo"
    print(eng.romanize("kata"))           # "kata"

    eng.save("my_lang_phonology.json")
    eng2 = PhonologyEngine.load("my_lang_phonology.json")
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Phoneme:
    """A single sound unit.

    symbol  : the canonical identifier (e.g. "p", "tʃ", "ŋ")
    kind    : "consonant" | "vowel"
    features: optional set of articulatory/acoustic tags
               e.g. {"voiced", "bilabial", "stop"}
    """
    symbol: str
    kind: str                        # "consonant" | "vowel"
    features: set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.kind not in ("consonant", "vowel"):
            raise ValueError(f"Phoneme kind must be 'consonant' or 'vowel', got {self.kind!r}")

    # JSON round-trip helpers
    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "kind": self.kind, "features": sorted(self.features)}

    @classmethod
    def from_dict(cls, d: dict) -> Phoneme:
        return cls(symbol=d["symbol"], kind=d["kind"], features=set(d.get("features", [])))


@dataclass
class SyllableTemplate:
    """A CV-pattern template such as "CVC" or "(C)V(C)".

    Notation:
      C  = required consonant slot
      V  = required vowel slot
      (C)= optional consonant slot
      (V)= optional vowel slot

    Parentheses mark optional positions.  The template is expanded into a list
    of slot specs: each slot is a dict {"kind": "C"|"V", "optional": bool}.
    """
    pattern: str   # e.g. "(C)V(C)"

    def __post_init__(self):
        # Validate that the pattern only contains C, V, (, )
        if not re.fullmatch(r"[CV()]+", self.pattern):
            raise ValueError(
                f"Invalid syllable template {self.pattern!r}. "
                "Use only C, V, (, )."
            )

    @property
    def slots(self) -> list[dict]:
        """Parse the pattern into a list of slot dicts."""
        result = []
        i = 0
        p = self.pattern
        while i < len(p):
            if p[i] == "(":
                # next char must be C or V, then )
                if i + 2 >= len(p) or p[i + 2] != ")":
                    raise ValueError(f"Malformed optional group at pos {i} in {p!r}")
                result.append({"kind": p[i + 1], "optional": True})
                i += 3
            elif p[i] in "CV":
                result.append({"kind": p[i], "optional": False})
                i += 1
            else:
                raise ValueError(f"Unexpected char {p[i]!r} in template {p!r}")
        return result

    def to_dict(self) -> dict:
        return {"pattern": self.pattern}

    @classmethod
    def from_dict(cls, d: dict) -> SyllableTemplate:
        return cls(pattern=d["pattern"])


@dataclass
class PhonotacticRule:
    """A constraint on which sound sequences are legal.

    Supported rule types
    --------------------
    no_cluster
        No more than `max_run` consecutive phonemes of the given class.
        e.g. {"type":"no_cluster","phoneme_class":"consonant","max_run":2}

    forbidden_sequence
        A specific sequence of phoneme symbols that may never occur together.
        e.g. {"type":"forbidden_sequence","sequence":["s","t","r"]}

    word_final_forbidden
        Phoneme(s) that may not appear at the end of a word.
        e.g. {"type":"word_final_forbidden","symbols":["h","ŋ"]}

    word_initial_forbidden
        Phoneme(s) that may not appear at the start of a word.
        e.g. {"type":"word_initial_forbidden","symbols":["ŋ"]}
    """
    rule_type: str
    params: dict = field(default_factory=dict)

    VALID_TYPES = {"no_cluster", "forbidden_sequence", "word_final_forbidden", "word_initial_forbidden"}

    def __post_init__(self):
        if self.rule_type not in self.VALID_TYPES:
            raise ValueError(f"Unknown rule type {self.rule_type!r}. Valid: {self.VALID_TYPES}")

    def to_dict(self) -> dict:
        return {"type": self.rule_type, **self.params}

    @classmethod
    def from_dict(cls, d: dict) -> PhonotacticRule:
        d = dict(d)
        rule_type = d.pop("type")
        return cls(rule_type=rule_type, params=d)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self):
        if self.valid:
            return f"Valid{(' — warnings: ' + '; '.join(self.warnings)) if self.warnings else ''}"
        return "Invalid — " + "; ".join(self.errors)


# ---------------------------------------------------------------------------
# Romanizer
# ---------------------------------------------------------------------------

class Romanizer:
    """Maps internal phoneme symbols to a romanization scheme.

    Any symbol not in the map is passed through unchanged.
    Multi-character phonemes (e.g. "tʃ") are matched before single-char ones,
    so longer matches always win.
    """

    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping: dict[str, str] = mapping or {}

    def romanize(self, word: str) -> str:
        """Convert a word written in phoneme symbols to romanized form."""
        if not self.mapping:
            return word
        # Sort by length descending so longer keys match first
        sorted_keys = sorted(self.mapping, key=len, reverse=True)
        result = []
        i = 0
        while i < len(word):
            for key in sorted_keys:
                if word[i:].startswith(key):
                    result.append(self.mapping[key])
                    i += len(key)
                    break
            else:
                result.append(word[i])
                i += 1
        return "".join(result)

    def to_dict(self) -> dict:
        return dict(self.mapping)

    @classmethod
    def from_dict(cls, d: dict) -> Romanizer:
        return cls(mapping=d)


# ---------------------------------------------------------------------------
# PhonologyEngine
# ---------------------------------------------------------------------------

class PhonologyEngine:
    """Top-level phonology manager for a conlang.

    Attributes
    ----------
    consonants  : list of Phoneme objects with kind="consonant"
    vowels      : list of Phoneme objects with kind="vowel"
    templates   : list of SyllableTemplate objects
    rules       : list of PhonotacticRule objects
    romanizer   : Romanizer instance
    """

    # IPA -> plain-ASCII near-equivalents used for auto-alias generation
    _IPA_ASCII_MAP = {
        "a": "ɑ", "o": "o", "ɒ": "o", "ɛ": "e", "ɪ": "i", "ʊ": "u",
        "ə": "uh", "ɕ": "sh", "ʑ": "zh", "ɾ": "r", "ʔ": "q",
        "ŋ": "ng", "ʃ": "sh", "ʒ": "zh", "θ": "th", "ð": "dh",
        "χ": "kh", "ʁ": "rh", "ħ": "hh", "ʕ": "ah", "ɣ": "gh",
        "β": "bh", "ɸ": "ph", "ʋ": "vh", "ɲ": "ny", "ʎ": "ly",
    }

    def __init__(
        self,
        consonants: list[Phoneme],
        vowels: list[Phoneme],
        templates: list[SyllableTemplate],
        rules: list[PhonotacticRule] | None = None,
        romanizer: Romanizer | None = None,
        input_aliases: dict[str, str] | None = None,
    ):
        if not consonants:
            raise ValueError("Must define at least one consonant.")
        if not vowels:
            raise ValueError("Must define at least one vowel.")
        if not templates:
            raise ValueError("Must define at least one syllable template.")

        self.consonants = consonants
        self.vowels = vowels
        self.templates = templates
        self.rules = rules or []
        self.romanizer = romanizer or Romanizer()

        # Lookup sets for fast membership tests
        self._consonant_set: set[str] = {p.symbol for p in consonants}
        self._vowel_set: set[str] = {p.symbol for p in vowels}
        self._all_phonemes: set[str] = self._consonant_set | self._vowel_set

        # Input aliases: typed shorthand -> canonical IPA symbol.
        # Auto-populated for IPA phonemes with plain-ASCII near-equivalents.
        self.input_aliases: dict[str, str] = {}
        self._build_default_aliases()
        if input_aliases:
            self.input_aliases.update(input_aliases)

    # ------------------------------------------------------------------
    # Input alias helpers
    # ------------------------------------------------------------------

    def _build_default_aliases(self) -> None:
        """Auto-generate ASCII aliases for IPA phonemes not already in inventory.

        For each IPA phoneme in the inventory that maps to an ASCII shorthand
        (via _IPA_ASCII_MAP), and where that shorthand is not itself a phoneme,
        register: ascii -> IPA so users can type normally.
        """
        # Build reverse map: IPA -> ASCII
        reverse = {v: k for k, v in self._IPA_ASCII_MAP.items()}
        for phoneme in self._all_phonemes:
            ascii_form = reverse.get(phoneme)
            if ascii_form and ascii_form not in self._all_phonemes:
                self.input_aliases[ascii_form] = phoneme

    def normalise_input(self, word: str) -> str:
        """Apply input aliases to a typed word before phonological processing.

        Longer aliases are matched first to avoid partial substitution.
        e.g. "nadh" -> "nɑdh" if alias a->ɑ is registered.
        """
        if not self.input_aliases:
            return word
        sorted_aliases = sorted(self.input_aliases, key=len, reverse=True)
        result = []
        i = 0
        while i < len(word):
            for alias in sorted_aliases:
                if word[i:].startswith(alias):
                    result.append(self.input_aliases[alias])
                    i += len(alias)
                    break
            else:
                result.append(word[i])
                i += 1
        return "".join(result)

    def add_alias(self, shorthand: str, phoneme: str) -> None:
        """Register a manual input alias. e.g. add_alias("r", "ɾ")"""
        if phoneme not in self._all_phonemes:
            raise ValueError(
                f"Phoneme {phoneme!r} is not in the inventory."
            )
        self.input_aliases[shorthand] = phoneme

    def list_aliases(self) -> str:
        """Return a formatted table of active input aliases."""
        if not self.input_aliases:
            return "  No input aliases defined."
        lines = ["  Input aliases (type \u2192 stored symbol):"]
        for k, v in sorted(self.input_aliases.items()):
            lines.append(f"    {k:<8} ->  {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> PhonologyEngine:
        """Build a PhonologyEngine from a plain dictionary.

        Minimal dict format::

            {
              "consonants": ["p", "t", "k"],   # symbols only (shorthand)
              "vowels":     ["a", "e", "i"],
              "syllable_templates": ["CV", "CVC"],
              "phonotactics": [
                  {"type": "no_cluster", "phoneme_class": "consonant", "max_run": 2}
              ],
              "romanization": {"k": "c"}
            }

        You can also pass full Phoneme dicts with features::

            "consonants": [{"symbol": "p", "kind": "consonant", "features": ["bilabial","stop"]}]
        """
        def _parse_phonemes(items: list, kind: str) -> list[Phoneme]:
            result = []
            for item in items:
                if isinstance(item, str):
                    result.append(Phoneme(symbol=item, kind=kind))
                else:
                    p = Phoneme.from_dict(item)
                    if p.kind != kind:
                        raise ValueError(
                            f"Phoneme {p.symbol!r} listed under '{kind}' but has kind={p.kind!r}"
                        )
                    result.append(p)
            return result

        consonants = _parse_phonemes(d.get("consonants", []), "consonant")
        vowels     = _parse_phonemes(d.get("vowels", []), "vowel")
        templates  = [SyllableTemplate.from_dict(t) if isinstance(t, dict) else SyllableTemplate(t)
                      for t in d.get("syllable_templates", [])]
        rules      = [PhonotacticRule.from_dict(r) for r in d.get("phonotactics", [])]
        romanizer     = Romanizer.from_dict(d.get("romanization", {}))
        input_aliases = d.get("input_aliases", {})

        return cls(consonants, vowels, templates, rules, romanizer, input_aliases)

    def to_dict(self) -> dict:
        return {
            "consonants": [p.to_dict() for p in self.consonants],
            "vowels":     [p.to_dict() for p in self.vowels],
            "syllable_templates": [t.to_dict() for t in self.templates],
            "phonotactics": [r.to_dict() for r in self.rules],
            "romanization": self.romanizer.to_dict(),
            "input_aliases": self.input_aliases,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the phonology definition to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Phonology saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> PhonologyEngine:
        """Load a phonology definition from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable summary of the phonology."""
        c_syms = " ".join(p.symbol for p in self.consonants)
        v_syms = " ".join(p.symbol for p in self.vowels)
        t_pats = "  ".join(t.pattern for t in self.templates)
        lines = [
            "=== Phonology Summary ===",
            f"Consonants ({len(self.consonants)}): {c_syms}",
            f"Vowels     ({len(self.vowels)}):     {v_syms}",
            f"Syllable templates: {t_pats}",
            f"Phonotactic rules:  {len(self.rules)}",
        ]
        for r in self.rules:
            lines.append(f"  • {r.rule_type}: {r.params}")
        return "\n".join(lines)

    def phoneme_kind(self, symbol: str) -> Optional[str]:
        """Return 'consonant', 'vowel', or None for unknown symbols."""
        if symbol in self._consonant_set:
            return "consonant"
        if symbol in self._vowel_set:
            return "vowel"
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_word(self, word: str) -> ValidationResult:
        """Check whether a word is legal under this phonology.

        Checks performed:
          1. Every symbol in the word belongs to the phoneme inventory.
          2. The word can be parsed into valid syllables (greedy).
          3. All phonotactic rules are satisfied.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not word:
            errors.append("Word is empty.")
            return ValidationResult(False, errors, warnings)

        # Apply input aliases before validation (e.g. a -> ɑ, sh -> ɕ)
        original = word
        word = self.normalise_input(word)
        if word != original:
            warnings.append(f"Normalised: {original!r} -> {word!r}")

        # 1. Unknown phonemes
        # Tokenise greedily by longest phoneme symbol first
        tokens = self._tokenize(word)
        if tokens is None:
            errors.append(
                f"Word {word!r} contains characters that don't match any phoneme symbol."
            )
            return ValidationResult(False, errors, warnings)

        unknown = [t for t in tokens if t not in self._all_phonemes]
        if unknown:
            errors.append(f"Unknown phoneme(s): {', '.join(unknown)}")

        # 2. Syllable parsing
        if not errors:
            syllables = self._parse_syllables(tokens)
            if syllables is None:
                errors.append(
                    f"Word {word!r} cannot be parsed into any combination of "
                    f"syllable templates ({[t.pattern for t in self.templates]})."
                )

        # 3. Phonotactic rules
        if not errors:
            for rule in self.rules:
                rule_errors = self._check_rule(tokens, rule)
                errors.extend(rule_errors)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def _tokenize(self, word: str) -> list[str] | None:
        """Split a word string into a list of phoneme symbol tokens.

        Uses greedy longest-match.  Returns None if any portion can't be matched.
        """
        all_symbols = sorted(self._all_phonemes, key=len, reverse=True)
        tokens = []
        i = 0
        while i < len(word):
            for sym in all_symbols:
                if word[i:].startswith(sym):
                    tokens.append(sym)
                    i += len(sym)
                    break
            else:
                return None   # unrecognised character
        return tokens

    def _parse_syllables(self, tokens: list[str]) -> list[list[str]] | None:
        """Try to parse tokens into syllables using the defined templates.

        Uses recursive backtracking. Returns a list of syllables (each a list
        of phoneme symbols), or None if parsing fails.
        """
        def _try(pos: int, acc: list) -> list | None:
            if pos == len(tokens):
                return acc
            for tmpl in self.templates:
                result = self._match_template(tokens, pos, tmpl)
                if result is not None:
                    syl, advance = result
                    found = _try(pos + advance, acc + [syl])
                    if found is not None:
                        return found
            return None

        return _try(0, [])

    def _match_template(
        self, tokens: list[str], pos: int, template: SyllableTemplate
    ) -> tuple[list[str], int] | None:
        """Try to match a syllable template starting at `pos`.

        Returns (matched_tokens, num_consumed) or None on failure.
        """
        slots = template.slots
        matched: list[str] = []
        i = pos

        for slot in slots:
            if i >= len(tokens):
                if slot["optional"]:
                    continue
                return None   # required slot, no token available
            kind_map = {"C": "consonant", "V": "vowel"}
            expected_kind = kind_map[slot["kind"]]
            actual_kind   = self.phoneme_kind(tokens[i])
            if actual_kind == expected_kind:
                matched.append(tokens[i])
                i += 1
            elif slot["optional"]:
                continue      # skip optional slot
            else:
                return None   # required slot mismatch

        if not matched:
            return None
        return matched, i - pos

    def _check_rule(self, tokens: list[str], rule: PhonotacticRule) -> list[str]:
        """Return a list of violation messages for one rule."""
        errors: list[str] = []
        p = rule.params

        if rule.rule_type == "no_cluster":
            cls_  = p.get("phoneme_class", "consonant")
            max_r = p.get("max_run", 2)
            run   = 0
            for tok in tokens:
                if self.phoneme_kind(tok) == cls_:
                    run += 1
                    if run > max_r:
                        errors.append(
                            f"Phonotactics violation: more than {max_r} "
                            f"consecutive {cls_}(s) near {tok!r}."
                        )
                        break
                else:
                    run = 0

        elif rule.rule_type == "forbidden_sequence":
            seq = p.get("sequence", [])
            if not seq:
                return errors
            n = len(seq)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i+n] == seq:
                    errors.append(
                        f"Phonotactics violation: forbidden sequence "
                        f"{' '.join(seq)!r} found."
                    )

        elif rule.rule_type == "word_final_forbidden":
            banned = set(p.get("symbols", []))
            if tokens and tokens[-1] in banned:
                errors.append(
                    f"Phonotactics violation: {tokens[-1]!r} is forbidden "
                    f"in word-final position."
                )

        elif rule.rule_type == "word_initial_forbidden":
            banned = set(p.get("symbols", []))
            if tokens and tokens[0] in banned:
                errors.append(
                    f"Phonotactics violation: {tokens[0]!r} is forbidden "
                    f"in word-initial position."
                )

        return errors

    # ------------------------------------------------------------------
    # Word generation
    # ------------------------------------------------------------------

    def random_syllable(self, template: SyllableTemplate | None = None) -> str:
        """Generate a random syllable from a template.

        If no template is given, one is chosen at random.
        """
        tmpl = template or random.choice(self.templates)
        result = []
        for slot in tmpl.slots:
            if slot["optional"] and random.random() < 0.5:
                continue   # skip optional slot ~50% of the time
            pool = self.consonants if slot["kind"] == "C" else self.vowels
            result.append(random.choice(pool).symbol)
        return "".join(result)

    def random_word(self, syllables: int = 2, max_attempts: int = 100) -> str:
        """Generate a random valid word with the given number of syllables.

        Retries up to `max_attempts` times if the generated word violates
        phonotactic rules.  Returns the best attempt on failure.
        """
        best = ""
        for _ in range(max_attempts):
            word = "".join(self.random_syllable() for _ in range(syllables))
            result = self.validate_word(word)
            if result.valid:
                return word
            best = word
        # Return last attempt with a warning rather than crash
        return best

    # ------------------------------------------------------------------
    # Romanization
    # ------------------------------------------------------------------

    def romanize(self, word: str) -> str:
        """Convert a word in internal phoneme notation to romanized form."""
        return self.romanizer.romanize(word)

    # ------------------------------------------------------------------
    # Interactive shell
    # ------------------------------------------------------------------

    def interactive_shell(self) -> None:  # pragma: no cover
        """Launch a simple REPL for exploring the phonology."""
        print("\n=== Conlang Phonology Shell ===")
        print(self.describe())
        print("\nCommands: validate <word> | generate [n_syllables] | romanize <word> | describe | quit\n")

        while True:
            try:
                raw = input("phonology> ").strip()
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

            elif cmd == "describe":
                print(self.describe())

            elif cmd == "aliases":
                print(self.list_aliases())

            elif cmd == "alias" and len(parts) >= 3:
                try:
                    self.add_alias(parts[1], parts[2])
                    print(f"  Alias added: {parts[1]!r} -> {parts[2]!r}")
                except ValueError as exc:
                    print(f"  ✗ {exc}")

            elif cmd == "validate" and len(parts) >= 2:
                word   = parts[1]
                result = self.validate_word(word)
                print(f"  {word!r}: {result}")

            elif cmd == "generate":
                n = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 2
                word = self.random_word(syllables=n)
                print(f"  Generated: {word!r}  →  romanized: {self.romanize(word)!r}")

            elif cmd == "romanize" and len(parts) >= 2:
                word = parts[1]
                print(f"  {word!r}  →  {self.romanize(word)!r}")

            else:
                print(f"  Unknown command or missing argument: {raw!r}")
                print("  Commands: validate <word> | generate [n] | romanize <word> | describe | quit")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A simple example language: Talevi
    talevi = PhonologyEngine.from_dict({
        "consonants": [
            {"symbol": "p", "kind": "consonant", "features": ["bilabial", "stop", "voiceless"]},
            {"symbol": "t", "kind": "consonant", "features": ["alveolar", "stop", "voiceless"]},
            {"symbol": "k", "kind": "consonant", "features": ["velar", "stop", "voiceless"]},
            {"symbol": "s", "kind": "consonant", "features": ["alveolar", "fricative", "voiceless"]},
            {"symbol": "n", "kind": "consonant", "features": ["alveolar", "nasal"]},
            {"symbol": "l", "kind": "consonant", "features": ["alveolar", "lateral"]},
            {"symbol": "v", "kind": "consonant", "features": ["labiodental", "fricative", "voiced"]},
        ],
        "vowels": [
            {"symbol": "a", "kind": "vowel", "features": ["open", "central"]},
            {"symbol": "e", "kind": "vowel", "features": ["mid", "front"]},
            {"symbol": "i", "kind": "vowel", "features": ["close", "front"]},
            {"symbol": "o", "kind": "vowel", "features": ["mid", "back"]},
            {"symbol": "u", "kind": "vowel", "features": ["close", "back"]},
        ],
        "syllable_templates": ["CV", "CVC", "V", "(C)V(C)"],
        "phonotactics": [
            {"type": "no_cluster", "phoneme_class": "consonant", "max_run": 2},
            {"type": "word_final_forbidden", "symbols": ["v"]},
            {"type": "forbidden_sequence", "sequence": ["s", "s"]},
        ],
        "romanization": {
            "k": "k",   # identity — could change to "c" for a different feel
            "s": "s",
            "v": "v",
        },
    })

    print(talevi.describe())
    print()

    # Validation tests
    tests = ["tani", "kata", "vvv", "ssane", "loke", "stank", ""]
    for w in tests:
        print(f"  validate({w!r}): {talevi.validate_word(w)}")
    print()

    # Word generation
    print("Random words:")
    for _ in range(8):
        word = talevi.random_word(syllables=random.randint(1, 3))
        print(f"  {word!r}  ({talevi.romanize(word)})")

    # Save / load round-trip
    talevi.save("/tmp/talevi_phonology.json")
    reloaded = PhonologyEngine.load("/tmp/talevi_phonology.json")
    assert reloaded.validate_word("tani").valid
    print("\nSave/load round-trip: OK")

    # Launch the REPL (comment out if running in CI)
    talevi.interactive_shell()
