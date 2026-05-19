# Conlang Workbench

A Python toolkit for building constructed languages (conlangs). Includes tools for phonology, vocabulary, grammar, and translating between your language and English, with emphasis on user input and mimicking natural languages.

Requires **Python 3.10+**.

---

## Quick Start

```bash
# Create a new language project
python main.py new mylang

# Open it in the interactive shell
python main.py open mylang

# One-shot translation without opening the shell
python main.py translate mylang "the water runs"
python main.py translate mylang "tani veka" --from-conlang

# Print a project summary and exit
python main.py info mylang
```

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point. Project wizard, top-level shell, CLI commands. |
| `phonology.py` | Sound system — phoneme inventory, syllable templates, phonotactics, input aliases. |
| `lexicon.py` | Vocabulary — dictionary manager, pattern analysis, irregularity annotation. |
| `morphology.py` | Word structure — inflection, paradigm tables, derivational morphology, morph rules. |
| `grammar.py` | Sentence structure — word order, agreement, negation, bidirectional translator. |

All five files go in the same directory. Each can also be imported as a standalone module if you want to use any layer independently.

---

## Project Structure

When you create a project, the workbench generates a folder containing four JSON files that persist your language between sessions:

```
mylang/
  project.json      ← name, description, file paths, settings
  phonology.json    ← phoneme inventory, syllable templates, phonotactics, aliases
  lexicon.json      ← all words, morphemes, and their metadata
  morphology.json   ← affix slots, feature map, morphophonological rules
  grammar.json      ← word order, agreement rules, negation, particles
```

These files can be edited by hand. The workbench saves automatically when you type `save` in any shell, or when you quit and confirm.

---

## The Four Layers

### 1. Phonology (`phonology.py`)

Defines the sound system of your language.

**Key features:**\
- **Phoneme inventory** — consonants and vowels, each with optional articulatory features (e.g. `{"bilabial", "stop", "voiceless"}`).\
- **Syllable templates** — CV patterns like `CV`, `CVC`, `CCVC`, `(C)V(C)`. Parentheses mark optional positions.\
- **Phonotactic rules** — constraints on sound sequences. Built-in rule types:\
  - `no_cluster` — max N consecutive consonants or vowels\
  - `forbidden_sequence` — a specific sequence that may never occur\
  - `word_final_forbidden` — sounds banned in word-final position\
  - `word_initial_forbidden` — sounds banned word-initially\
- **Input aliases** — map plain ASCII typing to IPA symbols automatically. For example, if your inventory contains `ɑ` (IPA alpha), typing `a` is automatically accepted and stored as `ɑ`. You can also define custom aliases like `sh → ɕ` or `r → ɾ`.\
- **Digraphs** — multi-character phoneme symbols (e.g. `dh` for `ð`) are fully supported. Longest-match tokenisation means `dh` always takes priority over `d` when the next character is `h`.\

**In the phonology shell:**
```
describe              Show full phonology summary
validate <word>       Check if a word is phonologically legal
generate [n]          Generate a random valid word (n syllables)
romanize <word>       Apply romanization map
aliases               List all input aliases
alias <from> <to>     Add a manual alias  e.g. alias r ɾ
quit
```

**Phonology validation is non-blocking for user-coined words.** If you add a word to the lexicon that breaks a rule you defined, you get a warning and are prompted for a reason, including in-universe loans or user overriding. The engine will not delete user inputed words.

---

### 2. Lexicon (`lexicon.py`)

The dictionary stores roots, affixes, and derived forms with full metadata.

**Entry fields:**

| Field | Description |
|---|---|
| `form` | The canonical written form. Affixes use hyphens: `-an`, `re-` |
| `gloss` | English translation or grammatical label |
| `pos` | Part of speech: `noun verb adjective adverb pronoun particle prefix suffix infix root other` |
| `source` | How the entry was created: `user generated imported derived` |
| `semantic_field` | Thematic grouping, e.g. `nature`, `kinship`, `body` |
| `tags` | Arbitrary labels, e.g. `["tense"]`, `["number"]` — used by morphology auto-loading |
| `notes` | Free-form user notes |
| `irregularity_reason` | Explanation for phonologically irregular forms (see below) |
| `related` | List of etymologically or semantically linked forms |

**Irregularity annotation** — a key feature. When a word breaks a phonotactic rule but the user keeps it anyway, the engine will ask: *"Why is this form irregular?"* The reason is stored alongside the warnings and displayed in lexicon listings, inflection traces, and the `irregular` command. This lets you document the historical or etymological logic behind any irregularities

```
⚠  = flagged, no reason given yet
✎  = flagged, reason documented
```

**In the lexicon shell:**
```
add <form> <gloss> [pos]       Add a user-coined entry
addb                           Add a batch (tab-separated: form gloss pos)
look <form>                    Full entry details
find <query>                   Search form, gloss, notes, tags
list [pos]                     List entries, optionally filtered by POS
irregular                      List all irregular entries with reasons
annotate <form>                Add/update irregularity reason for a flagged word
report                         Phonological pattern report across the lexicon
suggest                        Pattern suggestions from user-coined entries
gen <gloss> [pos] [syllables]  Generate a candidate word interactively
save <path>                    Save to JSON
load <path>                    Merge from JSON
quit
```

**Pattern analysis** — `report` and `suggest` scan your existing vocabulary and surface regularities: onset distributions, vowel nuclei, common bigrams, recurring sequences. `suggest` makes concrete recommendations: *"80% of your nouns end in a vowel — consider formalising that as a noun class marker."*

---

### 3. Morphology (`morphology.py`)

Handles word-internal structure and how roots combine with affixes to produce inflected or derived forms.

**Key features:**\
- **Feature map** — maps grammatical category+value pairs to affix forms. e.g. `tense=past → -an`, `number=plural → -i`. Auto-populated from lexicon entries tagged with `pos=suffix/prefix` and a `tags` list.\
- **Affix slots** — define the ordered positions of affixes for each POS. e.g. for verbs: `aspect (prefix, pos=0) → root → tense (suffix, pos=0) → number (suffix, pos=1)`. Slot ordering determines the final shape of complex word forms.\
- **Morphophonological rules** — sound changes at morpheme boundaries. Built-in operations:\
  - `delete_stem_final` — drop the last character of the stem\
  - `delete_affix_initial` — drop the first character of the affix\
  - `insert` — insert a string at the boundary\
  - `replace_stem_final` / `replace_affix_initial` — substitution\
  - `geminate_boundary` — double the consonant at the junction\
- **Paradigm builder** — `build_paradigm()` generates a full inflection table for a root across a list of feature bundles.\
- **Derivation** — `derive()` creates new lexemes from existing roots using derivational affixes, and optionally adds them to the lexicon with `source="derived"`.\
- **Morphological analysis** — `analyse()` runs in reverse: strips known affixes from a surface form and returns candidate root+feature parses.\

**Irregularity tracing** — if a root has a documented irregularity reason in the lexicon, it appears inline in the inflection trace.

**In the morphology shell:**
```
inflect <root> <cat=val> ...    Inflect a root  e.g. inflect tani number=plural
trace <root> <cat=val> ...      Inflect with full derivation log
analyse <word>                  Morphological parse (reverse)
paradigm <root> [pos]           Full inflection table
features                        Show all mapped feature categories
slots                           Show affix slot templates
derive <root> <affix> <gloss>   Create a derived word
map <category> <value> <affix>  Add a feature mapping  e.g. map tense past -an
loadlex                         Auto-load affixes from lexicon
save / load <path>
quit
```

---

### 4. Grammar & Translation (`grammar.py`)

Assembles inflected tokens into sentences and translates bidirectionally.

**Grammar features:**\
- **Word order** — any of the six standard typologies: `SVO SOV VSO VOS OVS OSV`. Set as a base typology; custom `WordOrderRule` objects can override it for specific clause types (e.g. interrogatives are verb-initial even in an otherwise SOV language).\
- **Agreement rules** — `AgreementRule(controller, target, feature)` copies a feature from one clause argument to another. e.g. verb agrees with subject in number.\
- **Case assignment** — maps clause roles to case values: `subject→nominative`, `direct_object→accusative`, etc. Feeds directly into the morphology pipeline.\
- **Negation** — configurable particle (`before_verb`, `after_verb`, `clause_final`) or verb affix.\
- **Particles** — arbitrary grammatical particles keyed by function: `question`, `topic`, `focus`, etc.\

**Translation pipeline (English → conlang):**\
1. Parse English sentence into subject / verb / object / modifiers, detecting tense and negation.\
2. Look up each lemma in the lexicon by English gloss.\
3. Inflect each token using the morphology engine with its grammatical features.\
4. Apply word-order template and agreement rules.\
5. Insert negation and particles.\

**Translation pipeline (conlang → English):**\
1. Tokenize the conlang sentence.\
2. Strip known affixes via morphological analysis → root + features.\
3. Look up root in lexicon → English gloss.\
4. Reconstruct English with basic tense marking.\

The `trace` command / `TranslationResult.log()` prints every step of either pipeline.

**In the grammar/translation shell:**
```
to <English sentence>           Translate English → conlang
from <conlang sentence>         Translate conlang → English
trace <English sentence>        Translate with full derivation log
batch                           Translate multiple sentences
phrasebook                      Build a formatted phrasebook
order <SVO|SOV|VSO|…>           Set word order typology
neg <particle> [position]       Set negation particle
particle <name> <form>          Add a grammatical particle
case <role> <case>              Set case for a clause role
agree <controller> <target> <feature>   Add agreement rule
override <english> <conlang>    Manual translation override
grammar                         Show current grammar settings
save / load <path>
quit
```

---

## Design Philosophy

**The user is the creative authority while using this tool.** This is a conlanging assist to help develop languages the way that natural languages do. 

- Words that break phonotactic rules are kept when `source="user"`, with a warning and a prompt for your reason.
- The `irregularity_reason` field lets you document the linguistic logic behind exceptions such as borrowings, fossilized forms, dialect remnants.
- Generated words always go through an accept / modify / skip prompt before entering the lexicon.
- Every translation and inflection can be traced step by step.
- The `suggest` command surfaces patterns in your own coinages rather than imposing a system, so the user can decide whether to systematize or keep the variation.

---

## Adding Words — Tips

**Affixes** are just lexicon entries with the right `pos` and `tags`:
```
form: "-an"   gloss: "past"   pos: "suffix"   tags: ["tense"]
form: "-i"    gloss: "plural" pos: "suffix"   tags: ["number"]
form: "re-"   gloss: "again"  pos: "prefix"   tags: ["aspect"]
```
Once these are in the lexicon, `morph.load_from_lexicon()` (called automatically on project open) picks them up and registers the feature mappings. No extra configuration needed.

**IPA input** — if your phoneme inventory uses IPA symbols, you don't need to type IPA when adding words. The alias system maps plain ASCII to your stored symbols automatically (e.g. `a → ɑ`, `sh → ɕ`). You can add custom aliases with `alias <from> <to>` in the phonology shell, or with `phonology.add_alias("r", "ɾ")` in code.

**Batch import** — you can maintain your vocabulary in a spreadsheet and import it:
```
# TSV format (tab-separated), with header row:
# form    gloss    pos    semantic_field    notes    tags
python -c "
from lexicon import Lexicon
from phonology import PhonologyEngine
phon = PhonologyEngine.load('mylang/phonology.json')
lex  = Lexicon.load('mylang/lexicon.json', phonology=phon)
lex.import_tsv('my_words.tsv')
lex.save('mylang/lexicon.json')
"
```

---

## Extending the Workbench

The five modules are designed to be imported independently. Some useful extension points:

**Custom phonotactic rule types** — add a new branch to `_check_rule()` in `phonology.py`. The rule dict is open-ended so no schema changes needed.

**Custom morphophonological operations** — add to `VALID_OPERATIONS` and a new branch in `_apply_rules()` in `morphology.py`.

**Custom word order rules** — instantiate `WordOrderRule` with any slot ordering and add it with `grammar.add_order_rule()`. The `applies_when` dict supports transitivity conditions; extend `_select_template()` for additional conditions.

**Suppletive/irregular inflected forms** (planned) — the natural next step is a per-cell override table in `MorphologyEngine`: `morph.add_irregular(root, features, surface_form, reason)`.

**Sound change engine** (planned) — a diachronic layer applying ordered phonological rules to the whole lexicon to simulate historical change or dialect variation.

---


## License

Feel free to use and adapt this code for your own conlanging needs!
