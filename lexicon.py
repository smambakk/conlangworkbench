"""
lexicon.py — Conlang Lexicon Manager
======================================
Stores, validates, and analyses the vocabulary of a constructed language.

Design philosophy
-----------------
Users are the creative authority. The lexicon never silently overwrites or
rejects a word the user has decided on — instead it:
  • Warns (not errors) when a word bends phonotactics, so the user can
    consciously decide to keep an irregular form.
  • Tracks *why* a word was added (user-coined vs generated vs imported),
    so the user knows which entries are "load-bearing" to their vision.
  • Surfaces phonological patterns across the lexicon so users can
    consciously extend or break them.
  • Supports morpheme entries (prefixes, suffixes, roots) alongside full
    words, so users can build a compositional vocabulary.

Components
----------
EntrySource   : enum-like tag for how an entry entered the lexicon
LexiconEntry  : a single dictionary entry (root or morpheme)
Lexicon       : main manager — add, search, analyse, import/export
LexiconShell  : interactive REPL for the lexicon

Usage
-----
    from phonology import PhonologyEngine
    from lexicon import Lexicon, LexiconEntry

    phon = PhonologyEngine.load("my_phonology.json")
    lex  = Lexicon(phon)

    # Add a word the user has already decided on
    lex.add("tani",  gloss="water",  pos="noun",   source="user")
    lex.add("veka",  gloss="to run", pos="verb",   source="user")
    lex.add("-an",   gloss="PAST",   pos="suffix",  source="user")

    # Search and analyse
    print(lex.lookup("tani"))
    print(lex.find_by_gloss("water"))
    print(lex.pattern_report())

    lex.save("my_lexicon.json")
    lex2 = Lexicon.load("my_lexicon.json", phon)
"""

from __future__ import annotations

import json
import re
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phonology import PhonologyEngine  # noqa: F401

try:
    from phonology import PhonologyEngine  # type: ignore
except ImportError:
    PhonologyEngine = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_POS = {
    "noun", "verb", "adjective", "adverb", "pronoun",
    "particle", "prefix", "suffix", "infix", "root", "other",
}

VALID_SOURCES = {
    "user",       # user explicitly coined this word/morpheme
    "generated",  # produced by the random generator
    "imported",   # loaded from an external word list
    "derived",    # built from existing roots + morphology rules
}


# ---------------------------------------------------------------------------
# LexiconEntry
# ---------------------------------------------------------------------------

@dataclass
class LexiconEntry:
    """A single lexicon entry — a root word, morpheme, or derived form.

    form        : the canonical written form (e.g. "tani", "-an", "re-")
                  Affixes use leading/trailing hyphens as a convention.
    gloss       : brief English translation or grammatical label
                  (e.g. "water", "PAST", "agent nominaliser")
    pos         : part of speech / morpheme type (see VALID_POS)
    source      : how this entry was created (see VALID_SOURCES)
    notes       : free-form notes from the user (etymology, usage, etc.)
    tags        : arbitrary string labels for grouping (e.g. "nature", "body")
    phonology_ok: True = passed phonology validation on entry
    warnings    : list of phonology warnings (entry kept despite warnings)
    semantic_field: broad thematic grouping (e.g. "environment", "kinship")
    related     : forms that are etymologically or semantically linked
    """
    form: str
    gloss: str
    pos: str                              = "other"
    source: str                           = "user"
    notes: str                            = ""
    tags: list[str]                       = field(default_factory=list)
    phonology_ok: bool                    = True
    warnings: list[str]                   = field(default_factory=list)
    irregularity_reason: str              = ""
    semantic_field: str                   = ""
    related: list[str]                    = field(default_factory=list)

    def __post_init__(self):
        if self.pos not in VALID_POS:
            raise ValueError(f"Unknown POS {self.pos!r}. Valid: {sorted(VALID_POS)}")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"Unknown source {self.source!r}. Valid: {sorted(VALID_SOURCES)}")

    # Bare form — strip leading/trailing hyphens for phonology checks
    @property
    def bare_form(self) -> str:
        return self.form.strip("-")

    @property
    def is_affix(self) -> bool:
        return self.form.startswith("-") or self.form.endswith("-")

    def short_repr(self) -> str:
        src_tag = f"[{self.source}]" if self.source != "user" else ""
        warn    = " ⚠" if self.warnings and not self.irregularity_reason else ""
        irreg   = " ✎" if self.irregularity_reason else ""
        return f"{self.form:<14} {self.pos:<12} '{self.gloss}'{warn}{irreg} {src_tag}"

    def full_repr(self) -> str:
        lines = [
            f"Form      : {self.form}",
            f"Gloss     : {self.gloss}",
            f"POS       : {self.pos}",
            f"Source    : {self.source}",
            f"Phon OK   : {self.phonology_ok}",
        ]
        if self.warnings:
            lines.append(f"Warnings  : {'; '.join(self.warnings)}")
        if self.irregularity_reason:
            lines.append(f"Irregular : {self.irregularity_reason}")
        if self.semantic_field:
            lines.append(f"Sem. field: {self.semantic_field}")
        if self.tags:
            lines.append(f"Tags      : {', '.join(self.tags)}")
        if self.related:
            lines.append(f"Related   : {', '.join(self.related)}")
        if self.notes:
            lines.append(f"Notes     : {self.notes}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "form": self.form,
            "gloss": self.gloss,
            "pos": self.pos,
            "source": self.source,
            "notes": self.notes,
            "tags": self.tags,
            "phonology_ok": self.phonology_ok,
            "warnings": self.warnings,
            "irregularity_reason": self.irregularity_reason,
            "semantic_field": self.semantic_field,
            "related": self.related,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LexiconEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

class Lexicon:
    """The vocabulary of a conlang.

    Parameters
    ----------
    phonology : PhonologyEngine — used to validate entries on add.
                Pass None to skip all phonological validation.
    """

    def __init__(self, phonology: Optional[PhonologyEngine] = None):
        self.phonology = phonology
        self._entries: dict[str, LexiconEntry] = {}   # form → entry

    # ------------------------------------------------------------------
    # Adding entries
    # ------------------------------------------------------------------

    def add(
        self,
        form: str,
        gloss: str,
        pos: str                   = "other",
        source: str                = "user",
        notes: str                 = "",
        tags: list[str]            = None,
        semantic_field: str        = "",
        related: list[str]         = None,
        irregularity_reason: str   = "",
        allow_phonology_warnings: bool = True,
        overwrite: bool            = False,
    ) -> LexiconEntry:
        """Add a new entry to the lexicon.

        Phonology is checked automatically if a PhonologyEngine is attached.
        If the word fails validation:
          - User-sourced words are kept with a warning (user's creative choice
            trumps generated rules).
          - Generated/imported words are rejected with an error.

        Parameters
        ----------
        allow_phonology_warnings : if False, treat phonology warnings as
            hard errors even for user-sourced words.
        overwrite : if True, replace an existing entry with the same form.
        """
        form = form.strip()
        if not form:
            raise ValueError("Entry form cannot be empty.")

        if form in self._entries and not overwrite:
            raise ValueError(
                f"Entry {form!r} already exists. Pass overwrite=True to replace it."
            )

        phon_ok   = True
        warnings  = []

        if self.phonology:
            bare = form.strip("-")
            result = self.phonology.validate_word(bare)
            if not result.valid:
                if source == "user" and allow_phonology_warnings:
                    # Respect the user's choice; record warnings
                    warnings = result.errors
                    phon_ok  = False
                    print(
                        f"  ⚠  {form!r} has phonology warnings (kept because source='user'):\n"
                        + "\n".join(f"     {e}" for e in result.errors)
                    )
                    # If no reason was pre-supplied, prompt interactively
                    if not irregularity_reason:
                        irregularity_reason = input(
                            "  Why is this form irregular? "
                            "(e.g. 'archaic borrowing', 'fossilised case ending') "
                            "[Enter to skip]: "
                        ).strip()
                else:
                    raise ValueError(
                        f"Entry {form!r} failed phonology validation:\n"
                        + "\n".join(f"  {e}" for e in result.errors)
                    )

        entry = LexiconEntry(
            form                = form,
            gloss               = gloss,
            pos                 = pos,
            source              = source,
            notes               = notes,
            tags                = list(tags or []),
            phonology_ok        = phon_ok,
            warnings            = warnings,
            irregularity_reason = irregularity_reason,
            semantic_field      = semantic_field,
            related             = list(related or []),
        )
        self._entries[form] = entry
        return entry

    def add_batch(self, entries: list[dict], **defaults) -> list[LexiconEntry]:
        """Add multiple entries from a list of dicts.

        Each dict should have at least 'form' and 'gloss'.  Any key absent
        from a dict falls back to `defaults`, then to the add() defaults.

        Example::

            lex.add_batch([
                {"form": "tani",  "gloss": "water",  "pos": "noun"},
                {"form": "veka",  "gloss": "to run", "pos": "verb"},
            ], source="user", semantic_field="environment")
        """
        results = []
        for d in entries:
            params = {**defaults, **d}
            try:
                e = self.add(**params)
                results.append(e)
            except ValueError as exc:
                print(f"  ✗  Skipped {d.get('form','?')!r}: {exc}")
        return results

    # ------------------------------------------------------------------
    # Lookup and search
    # ------------------------------------------------------------------

    def lookup(self, form: str) -> Optional[LexiconEntry]:
        """Return the entry for an exact form, or None."""
        return self._entries.get(form)

    def find_by_gloss(self, query: str, fuzzy: bool = True) -> list[LexiconEntry]:
        """Search entries by gloss text.

        If fuzzy=True, returns entries whose gloss contains the query
        (case-insensitive substring match).  If fuzzy=False, exact match only.
        """
        q = query.lower()
        if fuzzy:
            return [e for e in self._entries.values() if q in e.gloss.lower()]
        return [e for e in self._entries.values() if e.gloss.lower() == q]

    def find_by_pos(self, pos: str) -> list[LexiconEntry]:
        return [e for e in self._entries.values() if e.pos == pos]

    def find_by_tag(self, tag: str) -> list[LexiconEntry]:
        return [e for e in self._entries.values() if tag in e.tags]

    def find_by_semantic_field(self, field: str) -> list[LexiconEntry]:
        return [e for e in self._entries.values() if e.semantic_field == field]

    def list_irregular(self) -> str:
        """Return a formatted table of all entries with phonology warnings."""
        irregular = [e for e in self._entries.values() if e.warnings]
        if not irregular:
            return "  No irregular entries."
        lines = [
            f"{'Form':<14} {'POS':<10} {'Gloss':<24} {'Reason'}",
            "-" * 72,
        ]
        for e in sorted(irregular, key=lambda x: x.form):
            reason = e.irregularity_reason or "(no reason given)"
            lines.append(f"{e.form:<14} {e.pos:<10} {e.gloss:<24} {reason}")
        lines.append(f"\n  {len(irregular)} irregular entr{'y' if len(irregular)==1 else 'ies'}.")
        return "\n".join(lines)

    def annotate_irregular(self, form: str, reason: str) -> bool:
        """Add or update the irregularity reason for an existing entry.

        Returns True if the entry was found and updated, False otherwise.
        """
        entry = self._entries.get(form)
        if entry is None:
            print(f"  No entry for {form!r}.")
            return False
        if not entry.warnings:
            print(f"  {form!r} has no phonology warnings — nothing to annotate.")
            return False
        entry.irregularity_reason = reason
        print(f"  ✎  {form!r} annotated: {reason!r}")
        return True

    def find_by_source(self, source: str) -> list[LexiconEntry]:
        return [e for e in self._entries.values() if e.source == source]

    def search(self, query: str) -> list[LexiconEntry]:
        """Broad search across form, gloss, notes, and tags."""
        q = query.lower()
        results = []
        for e in self._entries.values():
            if (q in e.form.lower() or q in e.gloss.lower()
                    or q in e.notes.lower() or any(q in t for t in e.tags)):
                results.append(e)
        return results

    @property
    def entries(self) -> list[LexiconEntry]:
        return list(self._entries.values())

    @property
    def size(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Pattern analysis  ← the "help users systematise their ideas" core
    # ------------------------------------------------------------------

    def pattern_report(self) -> str:
        """Analyse the lexicon and surface phonological / structural patterns.

        Surfaces:
          • Onset consonant distribution
          • Coda consonant distribution (for CVC words)
          • Vowel nucleus distribution
          • Common initial and final bigrams
          • Average word length (syllables approximated by vowel count)
          • POS breakdown
          • Source breakdown
          • Warnings summary
        """
        if not self._entries:
            return "Lexicon is empty."

        words = [e for e in self._entries.values() if not e.is_affix]
        affixes = [e for e in self._entries.values() if e.is_affix]

        onsets:  Counter = Counter()
        codas:   Counter = Counter()
        nuclei:  Counter = Counter()
        bigrams: Counter = Counter()
        lengths: list[int] = []

        vowel_syms = set()
        cons_syms  = set()
        if self.phonology:
            vowel_syms = {p.symbol for p in self.phonology.vowels}
            cons_syms  = {p.symbol for p in self.phonology.consonants}

        for entry in words:
            f = entry.bare_form
            if not f:
                continue
            lengths.append(len(f))

            # Approximate onset = leading consonants
            i = 0
            onset = []
            while i < len(f) and f[i] in cons_syms:
                onset.append(f[i])
                i += 1
            if onset:
                onsets["".join(onset)] += 1
            else:
                onsets["(no onset)"] += 1

            # Approximate coda = trailing consonants
            j = len(f) - 1
            coda = []
            while j >= 0 and f[j] in cons_syms:
                coda.insert(0, f[j])
                j -= 1
            if coda:
                codas["".join(coda)] += 1
            else:
                codas["(open)"] += 1

            # Nuclei (all vowels)
            for ch in f:
                if ch in vowel_syms:
                    nuclei[ch] += 1

            # Character bigrams
            for k in range(len(f) - 1):
                bigrams[f[k:k+2]] += 1

        # POS and source breakdown
        pos_counts = Counter(e.pos for e in self._entries.values())
        src_counts = Counter(e.source for e in self._entries.values())
        warned     = [e.form for e in self._entries.values() if e.warnings]

        avg_len = sum(lengths) / len(lengths) if lengths else 0

        def top(counter: Counter, n: int = 5) -> str:
            return "  ".join(f"{k}×{v}" for k, v in counter.most_common(n))

        lines = [
            "=== Lexicon Pattern Report ===",
            f"Total entries  : {self.size}  ({len(words)} words, {len(affixes)} affixes)",
            f"Avg word length: {avg_len:.1f} chars",
            "",
            "— Onset consonants (most common) —",
            f"  {top(onsets)}",
            "",
            "— Vowel nuclei (most common) —",
            f"  {top(nuclei)}",
            "",
            "— Coda consonants (most common) —",
            f"  {top(codas)}",
            "",
            "— Common bigrams —",
            f"  {top(bigrams, 8)}",
            "",
            "— Parts of speech —",
            "  " + "  ".join(f"{k}:{v}" for k, v in pos_counts.most_common()),
            "",
            "— Entry sources —",
            "  " + "  ".join(f"{k}:{v}" for k, v in src_counts.most_common()),
        ]

        if warned:
            lines += ["", f"— Phonology warnings ({len(warned)}) —",
                      "  " + ", ".join(warned)]

        return "\n".join(lines)

    def suggest_pattern(self) -> str:
        """Look at user-coined entries and suggest regularities to systematise.

        This is a lightweight heuristic — it notices things like:
          • "Most of your nouns end in a vowel — you could make that a rule."
          • "Your user-added verbs tend to start with a consonant cluster."
          • "You have several words with the 'an' sequence — candidate for a morpheme?"
        """
        user_entries = self.find_by_source("user")
        if len(user_entries) < 3:
            return "Add at least 3 user-coined entries to get pattern suggestions."

        suggestions = []

        # --- Noun endings ---
        nouns = [e for e in user_entries if e.pos == "noun"]
        if len(nouns) >= 3:
            vowel_syms = {p.symbol for p in self.phonology.vowels} if self.phonology else set("aeiou")
            vowel_final = sum(1 for n in nouns if n.bare_form and n.bare_form[-1] in vowel_syms)
            pct = vowel_final / len(nouns)
            if pct >= 0.6:
                suggestions.append(
                    f"  {int(pct*100)}% of your nouns end in a vowel — consider making "
                    f"vowel-final a noun class marker."
                )
            cons_final = len(nouns) - vowel_final
            if cons_final / len(nouns) >= 0.6:
                suggestions.append(
                    f"  {int(cons_final/len(nouns)*100)}% of your nouns end in a consonant — "
                    f"you might use coda consonant as a noun marker."
                )

        # --- Verb onsets ---
        verbs = [e for e in user_entries if e.pos == "verb"]
        cons_syms = {p.symbol for p in self.phonology.consonants} if self.phonology else set("bcdfghjklmnpqrstvwxyz")
        if len(verbs) >= 3:
            cluster_start = sum(1 for v in verbs
                                if len(v.bare_form) >= 2
                                and v.bare_form[0] in cons_syms
                                and v.bare_form[1] in cons_syms)
            if cluster_start / len(verbs) >= 0.5:
                suggestions.append(
                    f"  Many of your verbs start with consonant clusters — "
                    f"maybe that's a verbal marker."
                )

        # --- Recurring sequences (candidate morphemes) ---
        # Count all bigrams and trigrams across user forms
        seq_count: Counter = Counter()
        for e in user_entries:
            f = e.bare_form
            for n in (2, 3):
                for i in range(len(f) - n + 1):
                    seq_count[f[i:i+n]] += 1

        threshold = max(2, len(user_entries) // 3)
        recurring = [(seq, cnt) for seq, cnt in seq_count.most_common(10)
                     if cnt >= threshold and len(seq) >= 2]
        if recurring:
            examples = ", ".join(f"{s!r} ({c}×)" for s, c in recurring[:4])
            suggestions.append(
                f"  Recurring sequences: {examples} — "
                f"any of these intentional morphemes?"
            )

        # --- Semantic field gaps ---
        fields = Counter(e.semantic_field for e in user_entries if e.semantic_field)
        if len(fields) < 2 and len(user_entries) >= 5:
            suggestions.append(
                "  Most entries lack a semantic_field tag — consider grouping "
                "them (e.g. 'nature', 'kinship', 'body') to spot gaps."
            )

        if not suggestions:
            return "No strong patterns detected yet — add more entries to surface regularities."

        return "=== Pattern Suggestions ===\n" + "\n".join(suggestions)

    # ------------------------------------------------------------------
    # Vocabulary generation helpers
    # ------------------------------------------------------------------

    def generate_word(
        self,
        gloss: str,
        pos: str        = "noun",
        syllables: int  = 2,
        tags: list[str] = None,
        semantic_field: str = "",
        add_to_lexicon: bool = True,
    ) -> Optional[LexiconEntry]:
        """Generate a phonologically valid word and optionally add it.

        The user is shown the candidate and asked to accept, modify, or
        reject it — preserving creative control.
        """
        if not self.phonology:
            raise RuntimeError("No PhonologyEngine attached; cannot generate words.")

        for _ in range(20):
            form = self.phonology.random_word(syllables=syllables)
            if form not in self._entries:
                break
        else:
            print("  Could not generate a unique form after 20 attempts.")
            return None

        print(f"\n  Candidate: {form!r}  (gloss: {gloss!r}, pos: {pos})")
        choice = input("  Accept [a], modify form [m], or skip [s]? ").strip().lower()

        if choice == "s":
            return None
        if choice == "m":
            form = input(f"  Enter your preferred form (was {form!r}): ").strip()
            if not form:
                return None

        if add_to_lexicon:
            return self.add(
                form=form, gloss=gloss, pos=pos,
                source="user" if choice == "m" else "generated",
                tags=tags or [], semantic_field=semantic_field,
            )
        return LexiconEntry(form=form, gloss=gloss, pos=pos)

    # ------------------------------------------------------------------
    # Import / export helpers
    # ------------------------------------------------------------------

    def import_tsv(
        self,
        path: str | Path,
        source: str      = "imported",
        delimiter: str   = "\t",
    ) -> int:
        """Import entries from a TSV/CSV file.

        Expected columns (header row required):
            form   gloss   pos   semantic_field   notes   tags

        Only 'form' and 'gloss' are mandatory.
        Returns the number of entries successfully added.
        """
        added = 0
        with open(path, encoding="utf-8") as f:
            header = [h.strip() for h in f.readline().split(delimiter)]
            for lineno, line in enumerate(f, 2):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(delimiter)]
                row   = dict(zip(header, parts))
                if "form" not in row or "gloss" not in row:
                    print(f"  Line {lineno}: missing form/gloss, skipping.")
                    continue
                tags = [t.strip() for t in row.get("tags", "").split(",")] if row.get("tags") else []
                try:
                    self.add(
                        form           = row["form"],
                        gloss          = row["gloss"],
                        pos            = row.get("pos", "other"),
                        source         = source,
                        notes          = row.get("notes", ""),
                        tags           = tags,
                        semantic_field = row.get("semantic_field", ""),
                    )
                    added += 1
                except ValueError as exc:
                    print(f"  Line {lineno}: {exc}")
        print(f"  Imported {added} entries from {path}")
        return added

    def export_tsv(self, path: str | Path) -> None:
        """Export all entries to a TSV file."""
        columns = ["form", "gloss", "pos", "source", "semantic_field",
                   "notes", "tags", "warnings", "irregularity_reason"]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\t".join(columns) + "\n")
            for e in self._entries.values():
                row = [
                    e.form,
                    e.gloss,
                    e.pos,
                    e.source,
                    e.semantic_field,
                    e.notes.replace("\n", " "),
                    ",".join(e.tags),
                    "; ".join(e.warnings),
                    e.irregularity_reason,
                ]
                f.write("\t".join(row) + "\n")
        print(f"  Exported {self.size} entries to {path}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self._entries.values()]}

    @classmethod
    def from_dict(cls, d: dict, phonology=None) -> Lexicon:
        lex = cls(phonology=phonology)
        for entry_d in d.get("entries", []):
            e = LexiconEntry.from_dict(entry_d)
            lex._entries[e.form] = e   # bypass validation on load
        return lex

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"Lexicon saved to {path} ({self.size} entries)")

    @classmethod
    def load(cls, path: str | Path, phonology=None) -> Lexicon:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, phonology=phonology)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def list_entries(
        self,
        pos: str        = None,
        source: str     = None,
        tag: str        = None,
        sem_field: str  = None,
        show_warnings: bool = True,
    ) -> str:
        """Return a formatted table of entries, with optional filters."""
        entries = self.entries
        if pos:
            entries = [e for e in entries if e.pos == pos]
        if source:
            entries = [e for e in entries if e.source == source]
        if tag:
            entries = [e for e in entries if tag in e.tags]
        if sem_field:
            entries = [e for e in entries if e.semantic_field == sem_field]

        if not entries:
            return "  (no entries match)"

        lines = [f"{'Form':<14} {'POS':<12} {'Gloss':<30} Source"]
        lines.append("-" * 70)
        for e in sorted(entries, key=lambda x: x.form):
            src  = f"[{e.source}]" if e.source != "user" else ""
            warn = " ⚠" if e.warnings else ""
            lines.append(f"{e.form:<14} {e.pos:<12} {e.gloss:<30} {src}{warn}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

class LexiconShell:
    """Interactive REPL for the lexicon manager."""

    HELP = """
Commands
--------
add  <form> <gloss> [pos]      Add a user-coined entry (pos defaults to 'noun')
addb                           Add a batch of entries interactively
look <form>                    Full detail for one entry
find <query>                   Search across form, gloss, notes, tags
pos  <pos>                     List all entries of a given POS
tag  <tag>                     List all entries with a tag
src  [source]                  List entries by source (user/generated/imported/derived)
gen  <gloss> [pos] [syllables] Generate a candidate word and optionally keep it
list                           List all entries
irregular                      List all phonologically irregular entries + reasons
annotate <form>                Add/update the irregularity reason for a flagged word
report                         Phonological pattern report
suggest                        Pattern suggestions based on user-coined entries
save <path>                    Save lexicon to JSON
load <path>                    Load lexicon from JSON (merges into current)
quit                           Exit
"""

    def __init__(self, lexicon: Lexicon):
        self.lex = lexicon

    def run(self):  # pragma: no cover
        print("\n=== Conlang Lexicon Shell ===")
        print(f"  {self.lex.size} entries loaded.")
        print("  Type 'help' for commands.\n")

        while True:
            try:
                raw = input("lexicon> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not raw:
                continue
            parts = raw.split(None, 3)
            cmd   = parts[0].lower()

            if cmd in ("quit", "exit", "q"):
                print("Goodbye.")
                break

            elif cmd == "help":
                print(self.HELP)

            elif cmd == "add" and len(parts) >= 3:
                form  = parts[1]
                gloss = parts[2]
                pos   = parts[3] if len(parts) > 3 else "noun"
                # Gather optional extras interactively
                notes = input("  Notes (optional, Enter to skip): ").strip()
                tags_raw = input("  Tags (comma-separated, Enter to skip): ").strip()
                sem   = input("  Semantic field (Enter to skip): ").strip()
                tags  = [t.strip() for t in tags_raw.split(",") if t.strip()]
                try:
                    e = self.lex.add(form=form, gloss=gloss, pos=pos,
                                     notes=notes, tags=tags, semantic_field=sem,
                                     source="user")
                    print(f"  ✓ Added: {e.short_repr()}")
                except ValueError as exc:
                    print(f"  ✗ {exc}")

            elif cmd == "addb":
                print("  Enter entries one per line as: form<TAB>gloss<TAB>pos")
                print("  (Empty line to finish)")
                rows = []
                while True:
                    line = input("  > ").strip()
                    if not line:
                        break
                    cols = line.split("\t")
                    if len(cols) < 2:
                        print("  Need at least form and gloss separated by TAB.")
                        continue
                    rows.append({"form": cols[0], "gloss": cols[1],
                                 "pos": cols[2] if len(cols) > 2 else "noun"})
                added = self.lex.add_batch(rows, source="user")
                print(f"  ✓ Added {len(added)} entries.")

            elif cmd == "look" and len(parts) >= 2:
                e = self.lex.lookup(parts[1])
                print(e.full_repr() if e else f"  No entry for {parts[1]!r}")

            elif cmd == "find" and len(parts) >= 2:
                results = self.lex.search(parts[1])
                if results:
                    for e in results:
                        print(f"  {e.short_repr()}")
                else:
                    print(f"  No results for {parts[1]!r}")

            elif cmd == "pos" and len(parts) >= 2:
                print(self.lex.list_entries(pos=parts[1]))

            elif cmd == "tag" and len(parts) >= 2:
                print(self.lex.list_entries(tag=parts[1]))

            elif cmd in ("src", "source"):
                src = parts[1] if len(parts) >= 2 else None
                print(self.lex.list_entries(source=src))

            elif cmd == "gen" and len(parts) >= 2:
                gloss = parts[1]
                pos   = parts[2] if len(parts) > 2 else "noun"
                syls  = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 2
                self.lex.generate_word(gloss=gloss, pos=pos, syllables=syls)

            elif cmd == "list":
                print(self.lex.list_entries())

            elif cmd == "irregular":
                print(self.lex.list_irregular())

            elif cmd == "annotate" and len(parts) >= 2:
                form   = parts[1]
                reason = input(f"  Reason for irregularity of {form!r}: ").strip()
                self.lex.annotate_irregular(form, reason)

            elif cmd == "report":
                print(self.lex.pattern_report())

            elif cmd == "suggest":
                print(self.lex.suggest_pattern())

            elif cmd == "save" and len(parts) >= 2:
                self.lex.save(parts[1])

            elif cmd == "load" and len(parts) >= 2:
                try:
                    loaded = Lexicon.load(parts[1], phonology=self.lex.phonology)
                    for e in loaded.entries:
                        self.lex._entries.setdefault(e.form, e)
                    print(f"  Merged. Total entries: {self.lex.size}")
                except Exception as exc:
                    print(f"  ✗ {exc}")

            else:
                print(f"  Unknown command: {raw!r}. Type 'help' for commands.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    try:
        from phonology import PhonologyEngine
        phon = PhonologyEngine.from_dict({
            "consonants": ["p","t","k","s","n","l","v","m","r"],
            "vowels":     ["a","e","i","o","u"],
            "syllable_templates": ["CV","CVC","V","(C)V(C)"],
            "phonotactics": [
                {"type": "no_cluster", "phoneme_class": "consonant", "max_run": 2},
                {"type": "word_final_forbidden", "symbols": ["v"]},
            ],
        })
    except ImportError:
        phon = None
        print("(phonology.py not found — running without phonology validation)\n")

    lex = Lexicon(phonology=phon)

    # User has already decided on some words for their language
    lex.add_batch([
        {"form": "tani",  "gloss": "water",       "pos": "noun",   "semantic_field": "nature"},
        {"form": "veka",  "gloss": "to run",       "pos": "verb",   "semantic_field": "motion"},
        {"form": "sola",  "gloss": "sun",          "pos": "noun",   "semantic_field": "nature"},
        {"form": "miru",  "gloss": "to see",       "pos": "verb",   "semantic_field": "perception"},
        {"form": "nale",  "gloss": "stone",        "pos": "noun",   "semantic_field": "nature"},
        {"form": "piko",  "gloss": "small",        "pos": "adjective"},
        {"form": "talev", "gloss": "sky",          "pos": "noun",   "semantic_field": "nature"},
        {"form": "-an",   "gloss": "PAST",         "pos": "suffix", "tags": ["tense"]},
        {"form": "-i",    "gloss": "PLURAL",       "pos": "suffix", "tags": ["number"]},
        {"form": "re-",   "gloss": "again/repeat", "pos": "prefix", "tags": ["aspect"]},
    ], source="user")

    print(lex.list_entries())
    print()
    print(lex.pattern_report())
    print()
    print(lex.suggest_pattern())
    print()

    # Save + reload round-trip
    lex.save("/tmp/demo_lexicon.json")
    lex2 = Lexicon.load("/tmp/demo_lexicon.json", phonology=phon)
    assert lex2.lookup("tani").gloss == "water"
    print("\nSave/load round-trip: OK")

    # Drop into the interactive shell
    shell = LexiconShell(lex)
    shell.run()
