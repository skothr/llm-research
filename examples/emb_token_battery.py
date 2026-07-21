"""Token battery for the embedding-atlas arc — committed data, no model deps.

Defines the curated word battery used to probe category structure in the
static input-embedding table of Qwen2.5-7B-Instruct (W_E, "layer 0"), plus the
aligned contrast PAIRS used for difference-direction analysis.

Design rules:
  * Every surface form appears in exactly ONE class (module-level assert —
    duplicated anchors bias centroid arithmetic; the nla-verbalizer arc
    learned this the hard way, see nla_vocab_atlas_capture.py's 2026-05-29
    dedup note).
  * Classes whose names match nla_vocab_atlas_capture.VOCAB keep that arc's
    member lists as a subset (country/capital/emotion are supersets; the
    layer-20 bridge intersects member lists per class, so supersets are safe).
  * Single-token coverage is NOT enforced here — emb_capture.py tests bare
    and leading-space variants per word against the tokenizer and reports
    drops. `bare_only` classes (punctuation, code, digits, CJK) skip the
    leading-space variant where it makes no linguistic sense.
  * PAIRS members must exist somewhere in the battery (asserted) but may live
    in any class — pairs are a relation over anchors, not a partition.

Run `python examples/emb_token_battery.py` for a tokenizer-free count report.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BatteryClass:
    """One anchor class: a named group of words sharing a category."""

    supergroup: str  # topic | function | punct | number | pos | sentiment |
    #                  register | code | morphology | multilingual
    words: tuple[str, ...]
    bare_only: bool = field(default=False)  # skip the leading-space variant


BATTERY: dict[str, BatteryClass] = {
    # ---- topic: nla-verbalizer overlap classes (supersets marked) ----------
    "country": BatteryClass(  # arc-1's 10 + 10 new
        "topic",
        (
            "France",
            "Germany",
            "Japan",
            "Brazil",
            "Egypt",
            "United Kingdom",
            "China",
            "Italy",
            "Russia",
            "India",
            "Spain",
            "Canada",
            "Turkey",
            "Mexico",
            "Poland",
            "Sweden",
            "Norway",
            "Greece",
            "Portugal",
            "Vietnam",
        ),
    ),
    "capital": BatteryClass(  # arc-1's 5 + 10 new
        "topic",
        (
            "Paris",
            "Berlin",
            "Tokyo",
            "London",
            "Madrid",
            "Rome",
            "Moscow",
            "Cairo",
            "Beijing",
            "Athens",
            "Vienna",
            "Oslo",
            "Dublin",
            "Lisbon",
            "Ottawa",
        ),
    ),
    "nature": BatteryClass(  # arc-1 verbatim
        "topic",
        (
            "spring",
            "summer",
            "autumn",
            "winter",
            "tree",
            "flower",
            "leaf",
            "butterfly",
            "snow",
            "mountain",
            "ocean",
            "sky",
        ),
    ),
    "codemath": BatteryClass(  # arc-1 verbatim
        "topic",
        ("function", "variable", "return", "equation", "integral", "theorem"),
    ),
    "emotion": BatteryClass(  # arc-1's 6 + 10 new
        "topic",
        (
            "happy",
            "sad",
            "anger",
            "fear",
            "love",
            "joy",
            "pride",
            "grief",
            "shame",
            "hope",
            "envy",
            "rage",
            "calm",
            "worry",
            "delight",
            "disgust",
        ),
    ),
    "refusal": BatteryClass("topic", ("refuse", "sorry", "cannot", "decline")),
    # ---- topic: new classes -------------------------------------------------
    "us_city": BatteryClass(
        "topic",
        (
            "Chicago",
            "Boston",
            "Seattle",
            "Houston",
            "Miami",
            "Denver",
            "Atlanta",
            "Dallas",
            "Phoenix",
            "Portland",
        ),
    ),
    "animal": BatteryClass(
        "topic",
        (
            "dog",
            "cat",
            "lion",
            "wolf",
            "bear",
            "tiger",
            "horse",
            "cow",
            "sheep",
            "pig",
            "mouse",
            "rabbit",
            "owl",
            "eagle",
            "snake",
            "fish",
            "whale",
            "shark",
            "bee",
            "ant",
            "spider",
            "frog",
            "deer",
            "fox",
            "elephant",
        ),
    ),
    "food": BatteryClass(
        "topic",
        (
            "bread",
            "cheese",
            "apple",
            "rice",
            "pizza",
            "egg",
            "coffee",
            "wine",
            "soup",
            "cake",
            "butter",
            "milk",
            "sugar",
            "salt",
            "meat",
            "pasta",
            "banana",
            "lemon",
        ),
    ),
    "color": BatteryClass(
        "topic",
        (
            "red",
            "blue",
            "green",
            "yellow",
            "purple",
            "pink",
            "brown",
            "black",
            "white",
            "gray",
        ),
    ),
    "body_part": BatteryClass(
        "topic",
        (
            "head",
            "hand",
            "eye",
            "heart",
            "brain",
            "bone",
            "finger",
            "arm",
            "leg",
            "foot",
            "ear",
            "nose",
            "mouth",
            "skin",
            "blood",
        ),
    ),
    "profession": BatteryClass(
        "topic",
        (
            "doctor",
            "teacher",
            "lawyer",
            "nurse",
            "soldier",
            "pilot",
            "judge",
            "farmer",
            "chef",
            "artist",
            "engineer",
            "scientist",
            "writer",
            "singer",
            "police",
        ),
    ),
    "family": BatteryClass(
        "topic",
        (
            "mother",
            "father",
            "sister",
            "brother",
            "son",
            "daughter",
            "uncle",
            "aunt",
            "cousin",
            "wife",
            "husband",
            "grandmother",
        ),
    ),
    "weather": BatteryClass(
        "topic",
        (
            "rain",
            "storm",
            "fog",
            "thunder",
            "wind",
            "hail",
            "frost",
            "mist",
            "breeze",
            "drizzle",
            "lightning",
            "cloud",
        ),
    ),
    "time_word": BatteryClass(
        "topic",
        (
            "today",
            "yesterday",
            "tomorrow",
            "morning",
            "evening",
            "night",
            "week",
            "year",
            "hour",
            "minute",
            "century",
            "decade",
            "noon",
            "midnight",
        ),
    ),
    "material": BatteryClass(
        "topic",
        (
            "wood",
            "metal",
            "glass",
            "stone",
            "steel",
            "gold",
            "silver",
            "plastic",
            "cotton",
            "leather",
            "copper",
            "iron",
        ),
    ),
    "vehicle": BatteryClass(
        "topic",
        (
            "car",
            "truck",
            "train",
            "plane",
            "boat",
            "bicycle",
            "bus",
            "ship",
            "motorcycle",
            "helicopter",
        ),
    ),
    "clothing": BatteryClass(
        "topic",
        (
            "shirt",
            "dress",
            "hat",
            "shoe",
            "coat",
            "scarf",
            "jacket",
            "sock",
            "glove",
            "belt",
        ),
    ),
    "sport": BatteryClass(
        "topic",
        (
            "soccer",
            "tennis",
            "golf",
            "chess",
            "boxing",
            "hockey",
            "baseball",
            "basketball",
            "rugby",
            "cricket",
        ),
    ),
    "month": BatteryClass(
        "topic",
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
    ),
    "day_of_week": BatteryClass(
        "topic",
        (
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ),
    ),
    "given_name": BatteryClass(
        "topic",
        ("Bill", "Frank", "Rose", "Mark", "Grace", "Jack"),
    ),
    "person": BatteryClass(
        "topic",
        (
            "man",
            "woman",
            "boy",
            "girl",
            "child",
            "baby",
            "adult",
            "person",
            "people",
            "human",
        ),
    ),
    "royal": BatteryClass(
        "topic",
        (
            "queen",
            "prince",
            "princess",
            "emperor",
            "duke",
            "lord",
            "knight",
            "throne",
            "crown",
            "palace",
        ),
    ),
    "religion": BatteryClass(
        "topic",
        (
            "god",
            "soul",
            "spirit",
            "heaven",
            "hell",
            "angel",
            "demon",
            "church",
            "temple",
            "prayer",
        ),
    ),
    "abstract": BatteryClass(
        "topic",
        (
            "freedom",
            "justice",
            "truth",
            "wisdom",
            "courage",
            "knowledge",
            "power",
            "peace",
            "faith",
            "honor",
            "liberty",
            "mercy",
        ),
    ),
    "landscape": BatteryClass(
        "topic",
        (
            "river",
            "lake",
            "forest",
            "desert",
            "valley",
            "island",
            "hill",
            "cliff",
            "beach",
            "cave",
            "canyon",
            "swamp",
        ),
    ),
    "instrument": BatteryClass(
        "topic",
        (
            "piano",
            "guitar",
            "violin",
            "drum",
            "flute",
            "trumpet",
            "cello",
            "harp",
            "banjo",
            "saxophone",
        ),
    ),
    "science": BatteryClass(
        "topic",
        (
            "physics",
            "chemistry",
            "biology",
            "atom",
            "molecule",
            "energy",
            "gravity",
            "electron",
            "cell",
            "gene",
        ),
    ),
    "tech": BatteryClass(
        "topic",
        (
            "computer",
            "internet",
            "software",
            "website",
            "email",
            "phone",
            "screen",
            "keyboard",
            "robot",
            "data",
        ),
    ),
    "language": BatteryClass(
        "topic",
        (
            "English",
            "French",
            "German",
            "Spanish",
            "Chinese",
            "Japanese",
            "Russian",
            "Italian",
            "Portuguese",
            "Vietnamese",
            "Greek",
            "Arabic",
            "Korean",
            "Hindi",
        ),
    ),
    # ---- function words: nla-verbalizer's 9 classes verbatim ---------------
    "article": BatteryClass("function", ("a", "an", "the")),
    "pronoun": BatteryClass("function", ("I", "you", "he", "she", "it", "we", "they")),
    "demonstrative": BatteryClass("function", ("this", "that", "these", "those")),
    "preposition": BatteryClass(
        "function",
        ("of", "in", "on", "at", "by", "with", "for", "from", "to", "into"),
    ),
    "conjunction": BatteryClass("function", ("and", "or", "but", "because", "if")),
    "auxiliary": BatteryClass(
        "function", ("is", "are", "was", "were", "has", "have", "will", "can")
    ),
    "negation": BatteryClass("function", ("not", "no", "never")),
    "quantifier": BatteryClass("function", ("all", "some", "every", "many", "few")),
    "wh_word": BatteryClass("function", ("what", "who", "where", "when", "why", "how")),
    # ---- punctuation: nla-verbalizer's 6 classes verbatim ------------------
    "p_ender": BatteryClass("punct", (".", "!", "?"), bare_only=True),
    "p_internal": BatteryClass("punct", (",", ";", ":"), bare_only=True),
    "p_quote": BatteryClass("punct", ('"', "'"), bare_only=True),
    "p_bracket": BatteryClass("punct", ("(", ")", "[", "]"), bare_only=True),
    "p_dash": BatteryClass("punct", ("-", "—"), bare_only=True),
    "p_special": BatteryClass("punct", ("/", "*", "#", "@"), bare_only=True),
    # ---- numbers & math: arc-1 verbatim + word/multi-digit extensions ------
    "number": BatteryClass(
        "number",
        ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
        bare_only=True,
    ),
    "math_op": BatteryClass("number", ("+", "="), bare_only=True),
    "number_word": BatteryClass(
        "number",
        (
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "hundred",
            "thousand",
            "million",
        ),
    ),
    "number_multi": BatteryClass(
        "number", ("42", "100", "1000", "2024", "3.14"), bare_only=True
    ),
    # ---- part-of-speech contrast --------------------------------------------
    "concrete_noun": BatteryClass(
        "pos",
        (
            "table",
            "chair",
            "door",
            "window",
            "book",
            "cup",
            "box",
            "wall",
            "road",
            "bridge",
        ),
    ),
    "action_verb": BatteryClass(
        "pos",
        (
            "run",
            "jump",
            "eat",
            "sleep",
            "swim",
            "fly",
            "sing",
            "dance",
            "push",
            "throw",
        ),
    ),
    "adjective": BatteryClass(
        "pos",
        (
            "big",
            "small",
            "fast",
            "slow",
            "hot",
            "cold",
            "tall",
            "bright",
            "dark",
            "heavy",
            "short",
            "light",
            "new",
            "old",
            "young",
            "rich",
            "poor",
            "strong",
            "weak",
            "deep",
        ),
    ),
    "speech_verb": BatteryClass(
        "pos",
        (
            "say",
            "tell",
            "speak",
            "shout",
            "whisper",
            "argue",
            "explain",
            "describe",
            "announce",
            "claim",
        ),
    ),
    "mental_verb": BatteryClass(
        "pos",
        (
            "think",
            "know",
            "believe",
            "remember",
            "forget",
            "understand",
            "imagine",
            "doubt",
            "guess",
            "learn",
        ),
    ),
    "adverb": BatteryClass(
        "pos",
        (
            "quickly",
            "slowly",
            "often",
            "rarely",
            "soon",
            "almost",
            "quite",
            "very",
            "always",
            "sometimes",
        ),
    ),
    # ---- sentiment / connotation (aligned via PAIRS "valence") -------------
    "positive": BatteryClass(
        "sentiment",
        (
            "good",
            "great",
            "excellent",
            "wonderful",
            "beautiful",
            "success",
            "win",
            "friend",
            "gift",
            "praise",
            "brave",
            "kind",
            "honest",
            "gentle",
            "pure",
        ),
    ),
    "negative": BatteryClass(
        "sentiment",
        (
            "bad",
            "terrible",
            "awful",
            "horrible",
            "ugly",
            "failure",
            "lose",
            "enemy",
            "poison",
            "insult",
            "cruel",
            "greedy",
            "dishonest",
            "harsh",
            "filthy",
        ),
    ),
    # ---- register (aligned via PAIRS "register") ----------------------------
    "formal": BatteryClass(
        "register",
        (
            "purchase",
            "commence",
            "inquire",
            "reside",
            "obtain",
            "assist",
            "utilize",
            "demonstrate",
            "terminate",
            "sufficient",
        ),
    ),
    "informal": BatteryClass(
        "register",
        ("buy", "begin", "ask", "live", "get", "help", "use", "show", "end", "enough"),
    ),
    # ---- code tokens (keywords avoid collisions with codemath/function) ----
    "code": BatteryClass(
        "code",
        (
            "def",
            "class",
            "import",
            "lambda",
            "elif",
            "async",
            "await",
            "null",
            "void",
            "bool",
            "struct",
            "enum",
            "printf",
            "println",
            "const",
            "static",
            "==",
            "+=",
            "->",
            "</",
            "{",
            "}",
            "//",
        ),
        bare_only=True,
    ),
    # ---- morphology (aligned via PAIRS "plural" / "case") ------------------
    "plural_form": BatteryClass(
        "morphology",
        (
            "dogs",
            "cats",
            "lions",
            "wolves",
            "trees",
            "flowers",
            "hands",
            "eyes",
            "bones",
            "fingers",
            "shirts",
            "shoes",
            "hats",
            "cars",
            "trucks",
            "trains",
            "boats",
            "years",
            "weeks",
            "books",
        ),
    ),
    "case_lower": BatteryClass(
        "morphology",
        (
            "may",
            "march",
            "august",
            "turkey",
            "china",
            "bill",
            "frank",
            "rose",
            "mark",
            "grace",
            "jack",
        ),
    ),
    "past_form": BatteryClass(
        "morphology",
        (
            "ran",
            "jumped",
            "ate",
            "slept",
            "swam",
            "flew",
            "sang",
            "danced",
            "pushed",
            "threw",
        ),
    ),
    # ---- multilingual concept groups (fr/es/de/zh/ja/ru where plausible) ----
    "xlat_water": BatteryClass(
        "multilingual", ("water", "eau", "agua", "Wasser", "水", "вода")
    ),
    "xlat_dog": BatteryClass(
        "multilingual", ("chien", "perro", "Hund", "cane", "犬", "狗", "собака")
    ),
    "xlat_house": BatteryClass(
        "multilingual", ("house", "maison", "casa", "Haus", "家", "дом")
    ),
    "xlat_king": BatteryClass(
        "multilingual", ("king", "roi", "rey", "König", "王", "король")
    ),
    "xlat_moon": BatteryClass(
        "multilingual", ("moon", "lune", "luna", "Mond", "月", "луна")
    ),
    "xlat_fire": BatteryClass(
        "multilingual", ("fire", "feu", "fuego", "Feuer", "火", "огонь")
    ),
}

# Aligned contrast pairs: (kind, a, b). Both members must exist in the battery
# (any class). Leading-space/bare twins are generated mechanically at capture
# time and are NOT listed here.
PAIRS: tuple[tuple[str, str, str], ...] = (
    # plural — singular lives in its semantic class, plural in plural_form
    ("plural", "dog", "dogs"),
    ("plural", "cat", "cats"),
    ("plural", "lion", "lions"),
    ("plural", "wolf", "wolves"),
    ("plural", "tree", "trees"),
    ("plural", "flower", "flowers"),
    ("plural", "hand", "hands"),
    ("plural", "eye", "eyes"),
    ("plural", "bone", "bones"),
    ("plural", "finger", "fingers"),
    ("plural", "shirt", "shirts"),
    ("plural", "shoe", "shoes"),
    ("plural", "hat", "hats"),
    ("plural", "car", "cars"),
    ("plural", "truck", "trucks"),
    ("plural", "train", "trains"),
    ("plural", "boat", "boats"),
    ("plural", "year", "years"),
    ("plural", "week", "weeks"),
    ("plural", "book", "books"),
    # case — lowercase common word vs capitalized proper-noun reading
    ("case", "may", "May"),
    ("case", "march", "March"),
    ("case", "august", "August"),
    ("case", "turkey", "Turkey"),
    ("case", "china", "China"),
    ("case", "bill", "Bill"),
    ("case", "frank", "Frank"),
    ("case", "rose", "Rose"),
    ("case", "mark", "Mark"),
    ("case", "grace", "Grace"),
    ("case", "jack", "Jack"),
    # valence — positive vs negative counterpart
    ("valence", "good", "bad"),
    ("valence", "beautiful", "ugly"),
    ("valence", "success", "failure"),
    ("valence", "win", "lose"),
    ("valence", "friend", "enemy"),
    ("valence", "praise", "insult"),
    ("valence", "kind", "cruel"),
    ("valence", "honest", "dishonest"),
    ("valence", "gentle", "harsh"),
    ("valence", "pure", "filthy"),
    # register — formal vs informal synonym
    ("register", "purchase", "buy"),
    ("register", "commence", "begin"),
    ("register", "inquire", "ask"),
    ("register", "reside", "live"),
    ("register", "obtain", "get"),
    ("register", "assist", "help"),
    ("register", "utilize", "use"),
    ("register", "demonstrate", "show"),
    ("register", "terminate", "end"),
    ("register", "sufficient", "enough"),
    # translation — English member of an xlat class vs a same-class translation
    # (within-xlat-class structure covers the rest; these anchor the analysis
    # on one high-resource language pair per concept)
    ("xlat", "water", "eau"),
    ("xlat", "dog", "chien"),
    ("xlat", "house", "casa"),
    ("xlat", "king", "rey"),
    ("xlat", "moon", "luna"),
    ("xlat", "fire", "fuego"),
    # gender — male vs female counterpart
    ("gender", "man", "woman"),
    ("gender", "boy", "girl"),
    ("gender", "king", "queen"),
    ("gender", "father", "mother"),
    ("gender", "brother", "sister"),
    ("gender", "uncle", "aunt"),
    ("gender", "husband", "wife"),
    ("gender", "son", "daughter"),
    ("gender", "prince", "princess"),
    ("gender", "he", "she"),
    # antonym — scalar-adjective opposites
    ("antonym", "big", "small"),
    ("antonym", "hot", "cold"),
    ("antonym", "fast", "slow"),
    ("antonym", "tall", "short"),
    ("antonym", "bright", "dark"),
    ("antonym", "heavy", "light"),
    ("antonym", "new", "old"),
    ("antonym", "rich", "poor"),
    ("antonym", "strong", "weak"),
    # past — verb base form vs simple past
    ("past", "run", "ran"),
    ("past", "jump", "jumped"),
    ("past", "eat", "ate"),
    ("past", "sleep", "slept"),
    ("past", "swim", "swam"),
    ("past", "fly", "flew"),
    ("past", "sing", "sang"),
    ("past", "dance", "danced"),
    ("past", "push", "pushed"),
    ("past", "throw", "threw"),
    # capital_of — country vs its capital city
    ("capital_of", "France", "Paris"),
    ("capital_of", "Germany", "Berlin"),
    ("capital_of", "Japan", "Tokyo"),
    ("capital_of", "China", "Beijing"),
    ("capital_of", "Russia", "Moscow"),
    ("capital_of", "Italy", "Rome"),
    ("capital_of", "Egypt", "Cairo"),
    ("capital_of", "Greece", "Athens"),
    ("capital_of", "Norway", "Oslo"),
    ("capital_of", "Portugal", "Lisbon"),
    ("capital_of", "Canada", "Ottawa"),
    ("capital_of", "Spain", "Madrid"),
    # lang_of — country vs its primary language name
    ("lang_of", "France", "French"),
    ("lang_of", "Germany", "German"),
    ("lang_of", "Spain", "Spanish"),
    ("lang_of", "China", "Chinese"),
    ("lang_of", "Japan", "Japanese"),
    ("lang_of", "Russia", "Russian"),
    ("lang_of", "Italy", "Italian"),
    ("lang_of", "Portugal", "Portuguese"),
    ("lang_of", "Vietnam", "Vietnamese"),
    ("lang_of", "Greece", "Greek"),
)

SUPERGROUPS: tuple[str, ...] = (
    "topic",
    "function",
    "punct",
    "number",
    "pos",
    "sentiment",
    "register",
    "code",
    "morphology",
    "multilingual",
)

# ---- module-level invariants (import-time, tokenizer-free) -----------------
_all_words = [w for c in BATTERY.values() for w in c.words]
assert len(_all_words) == len(set(_all_words)), (
    f"BATTERY has duplicates: {sorted({w for w in _all_words if _all_words.count(w) > 1})}"
)
_word_set = set(_all_words)
for _kind, _a, _b in PAIRS:
    assert _a in _word_set, f"PAIRS member {_a!r} ({_kind}) not in battery"
    assert _b in _word_set, f"PAIRS member {_b!r} ({_kind}) not in battery"
for _name, _cls in BATTERY.items():
    assert _cls.supergroup in SUPERGROUPS, (
        f"class {_name!r} has unknown supergroup {_cls.supergroup!r}"
    )


def main() -> None:
    per_super: dict[str, list[tuple[str, int]]] = {}
    for name, cls in BATTERY.items():
        per_super.setdefault(cls.supergroup, []).append((name, len(cls.words)))
    total = 0
    for sg in SUPERGROUPS:
        classes = per_super.get(sg, [])
        n = sum(c for _, c in classes)
        total += n
        listing = ", ".join(f"{name}({c})" for name, c in classes)
        print(f"{sg:13s} {n:4d}  {listing}")
    print(f"{'TOTAL':13s} {total:4d}  ({len(BATTERY)} classes)")
    kinds: dict[str, int] = {}
    for kind, _, _ in PAIRS:
        kinds[kind] = kinds.get(kind, 0) + 1
    print(
        f"{'PAIRS':13s} {len(PAIRS):4d}  "
        + ", ".join(f"{k}({v})" for k, v in kinds.items())
    )


if __name__ == "__main__":
    main()
