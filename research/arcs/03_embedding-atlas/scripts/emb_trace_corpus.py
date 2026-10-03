"""Scaled probe corpus for the tracing phase — committed data, no deps.

Extends the 14 pilot probes (emb_trace_capture.PROBES) to ~58 texts across
the same categories plus clause-/sentence-boundary-rich prose, longer lists
with varying inter-comma offsets (relevant to prediction P1d), multilingual
delimiters, and delimiter-free controls. Each entry: (id, category, text).
"""

from __future__ import annotations

PROBES_EXT: tuple[tuple[str, str, str], ...] = (
    # ---- comma lists, varying item lengths (inter-comma offset diversity) ----
    (
        "list_colors",
        "comma_list",
        "The flag used red, gold, green, white, and blue in narrow bands.",
    ),
    (
        "list_animals",
        "comma_list",
        "At the shelter we saw dogs, cats, rabbits, two parrots, an old turtle, and a goat.",
    ),
    (
        "list_tools_long",
        "comma_list",
        "The workshop drawer held a torque wrench, three sizes of hex keys, a stripped-down multimeter, spare fuses, heat-shrink tubing, and a label maker.",
    ),
    (
        "list_cities",
        "comma_list",
        "Flights leave daily for Tokyo, Berlin, Madrid, Cairo, Toronto, Oslo, and Lima.",
    ),
    (
        "list_verbs",
        "comma_list",
        "All afternoon they sorted, labeled, packed, weighed, stacked, and shipped the orders.",
    ),
    (
        "list_nested",
        "comma_list",
        "Bring warm clothes, that is, gloves, a wool hat, thick socks, and a scarf, plus your passport.",
    ),
    ("list_short_items", "comma_list", "Add salt, oil, rice, peas, mint, and lime."),
    (
        "list_long_items",
        "comma_list",
        "The course covers linear algebra with applications, probability theory from first principles, convex optimization methods, and large-scale numerical computing.",
    ),
    (
        "list_semicolons",
        "semicolon_list",
        "Three teams presented: the database group, who rebuilt the index; the network group, who cut latency; and the UI group, who simplified onboarding.",
    ),
    (
        "list_numbers_units",
        "comma_list",
        "The readings were 12.5 volts, 3.3 volts, 240 ohms, 18 milliamps, and 6 watts.",
    ),
    # ---- sentence- and clause-boundary-rich prose -----------------------------
    (
        "prose_three_sent",
        "prose",
        "The harbor opened at dawn. Fishing boats left in single file. By noon the docks were quiet again.",
    ),
    (
        "prose_four_sent",
        "prose",
        "Snow fell through the night. The roads stayed empty. Schools opened late. Nobody complained.",
    ),
    (
        "prose_subclauses",
        "prose",
        "When the results arrived, which took nearly a month, the committee, despite earlier doubts, approved the proposal.",
    ),
    (
        "prose_quote",
        "prose",
        '"We leave at six," she said, "and we are not waiting for anyone."',
    ),
    (
        "prose_parens",
        "prose",
        "The estimate (revised twice, most recently in March) still excluded shipping costs.",
    ),
    (
        "prose_colon",
        "prose",
        "One rule mattered above all: never deploy on a Friday afternoon.",
    ),
    (
        "prose_no_delim",
        "control_nodelim",
        "The tall pines along the northern ridge swayed slowly under the steady evening wind",
    ),
    (
        "prose_no_delim2",
        "control_nodelim",
        "Several young engineers from the southern plant reviewed the updated safety manual together",
    ),
    # ---- code with diverse delimiters -----------------------------------------
    (
        "code_dict",
        "code",
        'config = {"host": "localhost", "port": 8080, "debug": False, "retries": 3}\n',
    ),
    (
        "code_args",
        "code",
        'plot(xs, ys, color="red", lw=2, alpha=0.5, label="run 1")\n',
    ),
    (
        "code_chain",
        "code",
        'result = df.groupby("city").agg({"sales": "sum"}).reset_index().sort_values("sales")\n',
    ),
    (
        "code_c_like",
        "code",
        "for (int i = 0; i < n; i++) { total += weights[i] * values[i]; }\n",
    ),
    # ---- CJK with native delimiters --------------------------------------------
    ("cjk_list2", "cjk", "书架上有小说、词典、地图册、旧杂志和三本笔记本。"),
    ("cjk_two_sent", "cjk", "火车准时到站。乘客们安静地下了车。"),
    ("cjk_mixed_punct", "cjk", "他说：“明天见。”然后关上了门。"),
    ("ja_list", "cjk", "かばんには本、ペン、地図、お弁当が入っています。"),
    # ---- newline / markdown structure -------------------------------------------
    (
        "newline_recipe",
        "newline",
        "Steps:\n1. Heat the pan\n2. Add the onions\n3. Stir for five minutes\n4. Add the rice\n",
    ),
    (
        "newline_csv",
        "newline",
        "name,age,city\nAda,36,London\nLin,29,Taipei\nNoor,41,Cairo\n",
    ),
    (
        "newline_paras",
        "newline",
        "The first draft was too long.\n\nThe second draft lost the argument.\n\nThe third one worked.\n",
    ),
    # ---- mixed / question forms ---------------------------------------------------
    (
        "qa_list_answer",
        "comma_list",
        "Which countries border France? Spain, Italy, Switzerland, Germany, Belgium, and Luxembourg border France.",
    ),
    (
        "dates_inline",
        "numbers",
        "The meetings are on March 3, April 17, June 9, and October 28.",
    ),
    (
        "times_inline",
        "numbers",
        "Trains depart at 6:15, 7:40, 9:05, and 11:30 in the morning.",
    ),
    (
        "address_like",
        "comma_list",
        "Send the package to 14 Harbor Lane, Building C, Floor 2, Helsinki, Finland.",
    ),
    (
        "citation_like",
        "prose",
        "The method appears in Smith et al., 2019, Section 4, and was refined by Chen, 2022.",
    ),
    # ---- longer mixed passages (more tokens per probe) ---------------------------
    (
        "long_mixed1",
        "mixed",
        "The expedition packed carefully: ropes, two stoves, freeze-dried meals, and a satellite phone. Weather reports promised three clear days. On the fourth morning, clouds rolled in from the west, and the team, after a short debate, turned back.",
    ),
    (
        "long_mixed2",
        "mixed",
        "Quarterly results were mixed. Hardware revenue grew 12 percent, driven by the new sensor line, while services fell 3 percent. The board discussed pricing, churn, hiring, and the delayed factory, then adjourned until Thursday.",
    ),
    (
        "long_mixed3",
        "mixed",
        "In the old archive they found maps, letters, train schedules, and a box of photographs. Most were ruined. A few, kept dry inside a tin, showed the village before the dam: the mill, the narrow bridge, the rows of poplars along the road.",
    ),
)


def all_probes() -> tuple[tuple[str, str, str], ...]:
    """Pilot probes + extension, deduplicated by id (import-time assert)."""
    from emb_trace_capture import PROBES

    combined = tuple(PROBES) + PROBES_EXT
    ids = [p[0] for p in combined]
    assert len(ids) == len(set(ids)), "duplicate probe ids"
    return combined


if __name__ == "__main__":
    probes = all_probes()
    from collections import Counter

    print(f"{len(probes)} probes: {Counter(p[1] for p in probes).most_common()}")
