"""
main.py — Conlang Workbench
============================
Single entry point for the full conlang tool stack.

Wires together:
  phonology.py   — sound system
  lexicon.py     — vocabulary
  morphology.py  — word-internal structure
  grammar.py     — sentence structure & translation

A conlang project lives in a single directory:
  my_lang/
    project.json        ← metadata + settings
    phonology.json      ← phoneme inventory, syllable templates, phonotactics
    lexicon.json        ← all words and morphemes
    morphology.json     ← affix slots, feature map, morph rules
    grammar.json        ← word order, agreement, negation, particles

Usage
-----
  # Start fresh
  python main.py new my_lang

  # Open an existing project
  python main.py open my_lang

  # One-shot translation (no interactive shell)
  python main.py translate my_lang "the water runs"
  python main.py translate my_lang "tani veka" --from-conlang
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy imports — each module is only loaded if present
# ---------------------------------------------------------------------------

def _import_stack():
    """Import all four modules from the same directory as main.py."""
    here = Path(globals().get("__file__", __file__ if "__file__" in dir() else ".")).parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from phonology  import PhonologyEngine
    from lexicon    import Lexicon
    from morphology import MorphologyEngine, AffixSlot, MorphRule
    from grammar    import GrammarEngine, NegationRule, AgreementRule, Translator
    return (PhonologyEngine, Lexicon, MorphologyEngine,
            AffixSlot, MorphRule, GrammarEngine, NegationRule, AgreementRule, Translator)


# ---------------------------------------------------------------------------
# Project file helpers
# ---------------------------------------------------------------------------

PROJECT_FILE = "project.json"

DEFAULT_PROJECT = {
    "name":        "My Conlang",
    "description": "",
    "version":     "0.1.0",
    "files": {
        "phonology":  "phonology.json",
        "lexicon":    "lexicon.json",
        "morphology": "morphology.json",
        "grammar":    "grammar.json",
    },
    "settings": {
        "word_order":        "SVO",
        "negation_particle": "na",
        "negation_position": "before_verb",
        "question_particle": "",
    },
}

DEFAULT_PHONOLOGY = {
    "consonants": ["p","t","k","s","n","l","m","r"],
    "vowels":     ["a","e","i","o","u"],
    "syllable_templates": ["CV","CVC","V"],
    "phonotactics": [
        {"type": "no_cluster", "phoneme_class": "consonant", "max_run": 2}
    ],
    "romanization": {},
}


def _project_dir(name: str) -> Path:
    return Path(name)


def _load_project_meta(project_dir: Path) -> dict:
    p = project_dir / PROJECT_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"No project.json found in {project_dir}. "
            f"Run: python main.py new {project_dir.name}"
        )
    with open(p) as f:
        return json.load(f)


def _save_project_meta(project_dir: Path, meta: dict) -> None:
    with open(project_dir / PROJECT_FILE, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Stack loader
# ---------------------------------------------------------------------------

def load_stack(project_dir: Path):
    """Load the full module stack from a project directory.

    Returns (phonology, lexicon, morph, grammar, translator, meta).
    Missing files are initialised to safe defaults so partially-built
    projects always open cleanly.
    """
    (PhonologyEngine, Lexicon, MorphologyEngine,
     AffixSlot, MorphRule, GrammarEngine,
     NegationRule, AgreementRule, Translator) = _import_stack()

    meta = _load_project_meta(project_dir)
    files = meta.get("files", DEFAULT_PROJECT["files"])
    settings = meta.get("settings", DEFAULT_PROJECT["settings"])

    # ---- Phonology ----
    phon_path = project_dir / files["phonology"]
    if phon_path.exists():
        phon = PhonologyEngine.load(phon_path)
        print(f"  ✓ Phonology loaded  ({len(phon.consonants)}C + {len(phon.vowels)}V)")
    else:
        phon = PhonologyEngine.from_dict(DEFAULT_PHONOLOGY)
        print("  ℹ Phonology: using defaults (run 'phonology' to customise)")

    # ---- Lexicon ----
    lex_path = project_dir / files["lexicon"]
    if lex_path.exists():
        lex = Lexicon.load(lex_path, phonology=phon)
        print(f"  ✓ Lexicon loaded    ({lex.size} entries)")
    else:
        lex = Lexicon(phonology=phon)
        print("  ℹ Lexicon: empty (add words with 'lexicon')")

    # ---- Morphology ----
    morph_path = project_dir / files["morphology"]
    if morph_path.exists():
        morph = MorphologyEngine.load(morph_path, lexicon=lex, phonology=phon)
        n = morph.load_from_lexicon()   # pick up any new affixes added since last save
        print(f"  ✓ Morphology loaded ({len(morph.feature_map)} categories, "
              f"{len(morph.rules)} rules)")
    else:
        morph = MorphologyEngine(lexicon=lex, phonology=phon)
        morph.load_from_lexicon()
        print("  ℹ Morphology: empty (define affixes with 'morphology')")

    # ---- Grammar ----
    gram_path = project_dir / files["grammar"]
    if gram_path.exists():
        gram = GrammarEngine.load(gram_path, morph=morph)
        print(f"  ✓ Grammar loaded    (word order: {gram.word_order})")
    else:
        gram = GrammarEngine(morph=morph,
                             word_order=settings.get("word_order","SVO"))
        neg_p = settings.get("negation_particle","na")
        neg_pos = settings.get("negation_position","before_verb")
        if neg_p:
            gram.set_negation(NegationRule(particle=neg_p, position=neg_pos))
        q = settings.get("question_particle","")
        if q:
            gram.set_particle("question", q)
        print(f"  ℹ Grammar: defaults (word order: {gram.word_order})")

    translator = Translator(grammar=gram, lexicon=lex, morph=morph)
    return phon, lex, morph, gram, translator, meta


def save_stack(project_dir: Path, phon, lex, morph, gram, meta: dict) -> None:
    """Save all layers back to their project files."""
    files = meta.get("files", DEFAULT_PROJECT["files"])
    phon.save(project_dir / files["phonology"])
    lex.save(project_dir / files["lexicon"])
    morph.save(project_dir / files["morphology"])
    gram.save(project_dir / files["grammar"])
    _save_project_meta(project_dir, meta)
    print(f"  ✓ Project '{meta['name']}' saved.")


# ---------------------------------------------------------------------------
# Project creation wizard
# ---------------------------------------------------------------------------

def wizard_new_project(project_dir: Path) -> None:
    """Interactive wizard for creating a new conlang project."""
    (PhonologyEngine, Lexicon, MorphologyEngine,
     AffixSlot, MorphRule, GrammarEngine,
     NegationRule, AgreementRule, Translator) = _import_stack()

    print("\n" + "="*60)
    print("  Welcome to the Conlang Workbench!")
    print("  Let's set up your new language project.")
    print("="*60 + "\n")

    # --- Basic info ---
    name = input("  Language name (e.g. Talevi): ").strip() or "My Conlang"
    desc = input("  Short description (optional): ").strip()

    # --- Phonology wizard ---
    print("\n--- Phonology ---")
    print("  You can enter phoneme symbols separated by spaces.")
    print("  IPA symbols work (e.g. p t k s n l ŋ tʃ), or use your own.")

    raw_cons = input("  Consonants [default: p t k s n l m r]: ").strip()
    consonants = raw_cons.split() if raw_cons else ["p","t","k","s","n","l","m","r"]

    raw_vow = input("  Vowels [default: a e i o u]: ").strip()
    vowels = raw_vow.split() if raw_vow else ["a","e","i","o","u"]

    print("\n  Syllable templates use C (consonant) and V (vowel).")
    print("  Parentheses mark optional positions: (C)V(C)")
    raw_tmpl = input("  Templates [default: CV CVC V]: ").strip()
    templates = raw_tmpl.split() if raw_tmpl else ["CV","CVC","V"]

    # --- Grammar wizard ---
    print("\n--- Grammar ---")
    print("  Word order typologies:")
    print("    SVO (English), SOV (Japanese/Turkish), VSO (Arabic/Irish)")
    print("    VOS, OVS, OSV")
    raw_wo = input("  Word order [default: SVO]: ").strip().upper()
    word_order = raw_wo if raw_wo in {"SVO","SOV","VSO","VOS","OVS","OSV"} else "SVO"

    neg_p = input("  Negation particle (e.g. 'na', 'vel') [default: na]: ").strip() or "na"
    print("  Negation position: before_verb | after_verb | clause_final")
    neg_pos = input("  [default: before_verb]: ").strip() or "before_verb"

    q_p = input("  Question particle (optional, e.g. 'ka'): ").strip()

    # --- Seed lexicon ---
    print("\n--- Starter vocabulary (optional) ---")
    print("  You can add some words now, or skip and use the lexicon shell later.")
    print("  Format: one word per line as  form<TAB>gloss<TAB>pos")
    print("  Example:  tani<TAB>water<TAB>noun")
    print("  Press Enter on a blank line when done.\n")
    seed_entries = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        parts = line.split("\t")
        if len(parts) >= 2:
            seed_entries.append({
                "form":  parts[0].strip(),
                "gloss": parts[1].strip(),
                "pos":   parts[2].strip() if len(parts) > 2 else "noun",
            })
        else:
            print("  (Use TAB to separate form, gloss, pos)")

    # --- Build and save ---
    project_dir.mkdir(parents=True, exist_ok=True)

    phon = PhonologyEngine.from_dict({
        "consonants": consonants,
        "vowels": vowels,
        "syllable_templates": templates,
        "phonotactics": [
            {"type":"no_cluster","phoneme_class":"consonant","max_run":2}
        ],
    })

    lex = Lexicon(phonology=phon)
    if seed_entries:
        added = lex.add_batch(seed_entries, source="user")
        print(f"\n  Added {len(added)} seed entries.")

    morph = MorphologyEngine(lexicon=lex, phonology=phon)
    morph.load_from_lexicon()

    gram = GrammarEngine(morph=morph, word_order=word_order)
    gram.set_negation(NegationRule(particle=neg_p, position=neg_pos))
    if q_p:
        gram.set_particle("question", q_p)

    meta = {
        "name":        name,
        "description": desc,
        "version":     "0.1.0",
        "files":       DEFAULT_PROJECT["files"],
        "settings": {
            "word_order":        word_order,
            "negation_particle": neg_p,
            "negation_position": neg_pos,
            "question_particle": q_p,
        },
    }

    save_stack(project_dir, phon, lex, morph, gram, meta)
    print(f"\n  Project '{name}' created in ./{project_dir}/")
    print("  Run: python main.py open " + str(project_dir))


# ---------------------------------------------------------------------------
# Main interactive shell
# ---------------------------------------------------------------------------

BANNER = r"""
  ╔═══════════════════════════════════════╗
  ║      C O N L A N G  W O R K B E N C H ║
  ╚═══════════════════════════════════════╝
"""

MAIN_HELP = """
Commands — top level
--------------------
  phonology          Enter phonology shell (sounds, syllables, phonotactics)
  lexicon            Enter lexicon shell (words, morphemes, pattern analysis)
  morphology         Enter morphology shell (inflection, paradigms, derivation)
  grammar            Enter grammar & translation shell (word order, translate)

  translate  <text>  Quick English → conlang translation
  from       <text>  Quick conlang → English translation
  trace      <text>  Translate with full derivation log

  info               Show project summary
  phrasebook         Build a formatted phrasebook
  save               Save all layers to project files
  help               Show this message
  quit               Exit (prompts to save)
"""


def run_main_shell(project_dir: Path) -> None:
    """Open a project and launch the top-level shell."""
    from phonology  import PhonologyEngine
    from lexicon    import Lexicon
    from morphology import MorphologyEngine, AffixSlot, MorphRule
    from grammar    import (GrammarEngine, NegationRule, AgreementRule,
                            Translator, GrammarShell)
    from lexicon    import LexiconShell         # type: ignore
    from morphology import MorphologyShell      # type: ignore
    from phonology  import PhonologyEngine      # type: ignore

    print(BANNER)
    print(f"  Opening project: {project_dir}/\n")

    phon, lex, morph, gram, translator, meta = load_stack(project_dir)

    print(f"\n  Welcome to '{meta['name']}'")
    if meta.get("description"):
        print(f"  {meta['description']}")
    print("\n  Type 'help' for commands.\n")

    unsaved = False

    while True:
        try:
            raw = input(f"[{meta['name']}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "quit"

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd   = parts[0].lower()
        rest  = parts[1] if len(parts) > 1 else ""

        # ---- Sub-shells ----
        if cmd == "phonology":
            # Lightweight phonology REPL inline (delegates to PhonologyEngine.interactive_shell)
            phon.interactive_shell()
            unsaved = True

        elif cmd == "lexicon":
            from lexicon import LexiconShell
            LexiconShell(lex).run()
            # Reload affixes into morphology after lexicon changes
            morph.load_from_lexicon()
            unsaved = True

        elif cmd == "morphology":
            from morphology import MorphologyShell
            MorphologyShell(morph).run()
            unsaved = True

        elif cmd == "grammar":
            from grammar import GrammarShell
            GrammarShell(translator).run()
            unsaved = True

        # ---- Quick translation ----
        elif cmd == "translate" and rest:
            r = translator.translate_to_conlang(rest)
            print(f"  {r.surface}")
            for w in r.warnings:
                print(f"  ⚠ {w}")

        elif cmd == "from" and rest:
            r = translator.translate_to_english(rest)
            print(f"  {r.surface}")
            for w in r.warnings:
                print(f"  ⚠ {w}")

        elif cmd == "trace" and rest:
            r = translator.translate_to_conlang(rest)
            print(r.log())

        # ---- Phrasebook ----
        elif cmd == "phrasebook":
            print("  Enter English phrases (blank line to finish):")
            phrases = []
            while True:
                s = input("  > ").strip()
                if not s:
                    break
                phrases.append(s)
            if phrases:
                print(translator.build_phrasebook(phrases))

        # ---- Project info ----
        elif cmd == "info":
            _print_info(meta, phon, lex, morph, gram)

        # ---- Save ----
        elif cmd == "save":
            save_stack(project_dir, phon, lex, morph, gram, meta)
            unsaved = False

        elif cmd == "help":
            print(MAIN_HELP)

        elif cmd in ("quit","exit","q"):
            if unsaved:
                yn = input("  You have unsaved changes. Save before quitting? [Y/n]: ").strip().lower()
                if yn != "n":
                    save_stack(project_dir, phon, lex, morph, gram, meta)
            print("  Goodbye!")
            break

        else:
            print(f"  Unknown command: {raw!r}. Type 'help'.")


def _print_info(meta, phon, lex, morph, gram) -> None:
    print(f"\n  ╔══ {meta['name']} ══")
    if meta.get("description"):
        print(f"  ║  {meta['description']}")
    print( f"  ╠══ Phonology")
    print( f"  ║   Consonants : {' '.join(p.symbol for p in phon.consonants)}")
    print( f"  ║   Vowels     : {' '.join(p.symbol for p in phon.vowels)}")
    print( f"  ║   Templates  : {' '.join(t.pattern for t in phon.templates)}")
    print( f"  ╠══ Lexicon")
    print( f"  ║   Entries    : {lex.size}")
    user_e = len(lex.find_by_source("user"))
    gen_e  = len(lex.find_by_source("generated"))
    print( f"  ║   By source  : {user_e} user-coined, {gen_e} generated")
    print( f"  ╠══ Morphology")
    cats = list(morph.feature_map.keys())
    print( f"  ║   Categories : {', '.join(cats) if cats else '(none yet)'}")
    print( f"  ║   Rules      : {len(morph.rules)}")
    print( f"  ╠══ Grammar")
    print( f"  ║   Word order : {gram.word_order}")
    nr = gram.negation_rule
    print( f"  ║   Negation   : {nr.particle!r} ({nr.position})" if nr else "  ║   Negation   : none")
    if gram.particles:
        print(f"  ║   Particles  : {gram.particles}")
    print( f"  ╚══")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Conlang Workbench — build and translate constructed languages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py new my_lang          Create a new project
              python main.py open my_lang         Open interactive shell
              python main.py translate my_lang "the water runs"
              python main.py translate my_lang "tani veka" --from-conlang
              python main.py info my_lang         Quick project summary
        """),
    )
    sub = parser.add_subparsers(dest="command")

    # new
    p_new = sub.add_parser("new",  help="Create a new conlang project")
    p_new.add_argument("project",  help="Project directory name")

    # open
    p_open = sub.add_parser("open", help="Open a project in the interactive shell")
    p_open.add_argument("project",  help="Project directory name")

    # translate
    p_tr = sub.add_parser("translate", help="Translate a single sentence")
    p_tr.add_argument("project",       help="Project directory name")
    p_tr.add_argument("sentence",      help="Sentence to translate")
    p_tr.add_argument("--from-conlang",action="store_true",
                      help="Translate conlang → English instead of English → conlang")
    p_tr.add_argument("--trace",       action="store_true",
                      help="Print full derivation log")

    # info
    p_info = sub.add_parser("info", help="Print a project summary and exit")
    p_info.add_argument("project",    help="Project directory name")

    args = parser.parse_args()

    if args.command == "new":
        project_dir = _project_dir(args.project)
        if (project_dir / PROJECT_FILE).exists():
            yn = input(
                f"  Project '{args.project}' already exists. Overwrite? [y/N]: "
            ).strip().lower()
            if yn != "y":
                print("  Aborted.")
                return
        wizard_new_project(project_dir)

    elif args.command == "open":
        project_dir = _project_dir(args.project)
        run_main_shell(project_dir)

    elif args.command == "translate":
        project_dir = _project_dir(args.project)
        here = Path(__file__).parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        phon, lex, morph, gram, translator, meta = load_stack(project_dir)
        if args.from_conlang:
            r = translator.translate_to_english(args.sentence)
        else:
            r = translator.translate_to_conlang(args.sentence)
        if args.trace:
            print(r.log())
        else:
            print(r.surface)
            for w in r.warnings:
                print(f"⚠ {w}", file=sys.stderr)

    elif args.command == "info":
        project_dir = _project_dir(args.project)
        here = Path(__file__).parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        phon, lex, morph, gram, translator, meta = load_stack(project_dir)
        _print_info(meta, phon, lex, morph, gram)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
