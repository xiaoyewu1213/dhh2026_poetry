from __future__ import annotations

import json
import importlib.util
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency


BASE = Path("/Users/xiaoye/Documents/New project")
DATA = BASE / "outputs" / "dhh26_full"
PERSON_DIR = BASE / "outputs" / "dhh26_full" / "person_name_variation"
GEOJSON = Path("/Users/xiaoye/Downloads/areas.geojson")
OUT_HTML = PERSON_DIR / "name_variant_interactive_area_viewer.html"
OUT_ALIAS_HTML = PERSON_DIR / "name_variant_maps_integrated_viewer.html"
WORD_ANALYSIS = DATA / "word_analysis.parquet"
PLACE_SCRIPT = BASE / "scripts" / "map_word_in_english_only_name_variants_area.py"
REGIONAL_FEATURE_CANDIDATES = DATA / "regional_feature_candidate_tokens_by_category.csv"
SEMANTIC_FEATURE_DIR = DATA / "semantic_category_features_standard_lemma_area"
SEMANTIC_FEATURE_COUNTS = SEMANTIC_FEATURE_DIR / "semantic_category_feature_standard_lemma_area_counts_for_map.csv"
SEMANTIC_FEATURE_VARIANTS = SEMANTIC_FEATURE_DIR / "semantic_category_feature_standard_lemma_variant_summary.csv"
SEMANTIC_FEATURE_SUMMARY = SEMANTIC_FEATURE_DIR / "semantic_category_feature_summary.csv"
SEMANTIC_FEATURE_MENTIONS = SEMANTIC_FEATURE_DIR / "semantic_category_feature_mentions_with_standard_lemma.csv"


GROUPS = [
    "Ilmarinen",
    "Väinämöinen",
    "Lemminkäinen",
    "Smith / Blacksmith",
    "Jesus / Christ",
    "Mary / Maria / Maarja / Mari",
    "John / Jaan / Jussi / Hans",
    "Peter / Pietari / Peeter",
    "Martin / Mart / Märt",
    "Catherine / Kadri / Katri",
    "Anna / Anne / Ann / Anni",
]

GROUP_SHORT = {
    "Smith / Blacksmith": "Smith/Blacksmith",
    "Jesus / Christ": "Jesus/Christ",
    "Mary / Maria / Maarja / Mari": "Mary/Maria/Mari",
    "John / Jaan / Jussi / Hans": "John/Jaan/Jussi/Hans",
    "Peter / Pietari / Peeter": "Peter/Pietari/Peeter",
    "Martin / Mart / Märt": "Martin/Mart/Märt",
    "Catherine / Kadri / Katri": "Catherine/Kadri/Katri",
    "Anna / Anne / Ann / Anni": "Anna/Anne/Anni",
    "Ilmarinen": "Ilmarinen",
    "Väinämöinen": "Väinämöinen",
    "Lemminkäinen": "Lemminkäinen",
}

SLUGS = {
    "Smith / Blacksmith": "smith_blacksmith",
    "Jesus / Christ": "jesus_christ",
    "Mary / Maria / Maarja / Mari": "mary_maria_mari",
    "John / Jaan / Jussi / Hans": "john_jaan_jussi_hans",
    "Peter / Pietari / Peeter": "peter_pietari_peeter",
    "Martin / Mart / Märt": "martin_mart_mart",
    "Catherine / Kadri / Katri": "catherine_kadri_katri",
    "Anna / Anne / Ann / Anni": "anna_anne_anni",
    "Ilmarinen": "ilmarinen",
    "Väinämöinen": "vainamoinen",
    "Lemminkäinen": "lemminkainen",
}

STANDARD_LABELS = {
    "Jesus / Christ": ["Jeesus", "Christ/Kristus", "Other"],
    "Mary / Maria / Maarja / Mari": ["Mari", "Maria", "Maija/Maie", "Maarja", "Maaria", "Other"],
    "John / Jaan / Jussi / Hans": ["Jaan", "Jussi/Juss", "Hans", "Ants", "Iivana", "Other"],
    "Peter / Pietari / Peeter": ["Pietari", "Peeter", "Pekka", "Petteri/Petri", "Other"],
    "Martin / Mart / Märt": ["Märt", "Mart/Martin", "Mard/Mardi", "Märti", "Martti", "Other"],
    "Catherine / Kadri / Katri": ["Kadri", "Katri", "Kaisa", "Kati", "Katariina/Katriina", "Other"],
    "Anna / Anne / Ann / Anni": ["Anne", "Anni", "Ann", "Anna", "Anu", "Other"],
    "Ilmarinen": ["Ilmarinen", "Other"],
    "Väinämöinen": ["Väinämöinen", "Other"],
    "Lemminkäinen": ["Lemminkäinen", "Other"],
}

ENGLISH_LABELS = {
    "Smith / Blacksmith": ["seppä", "sepp", "compound -sepp", "seppo", "takoja", "Other"],
    "Jesus / Christ": ["Jesus", "Christ", "Other"],
    "Mary / Maria / Maarja / Mari": ["Mary", "Mari", "Maija/Maie", "Maria/Maaria", "Other"],
    "John / Jaan / Jussi / Hans": ["Jussi", "Jaan/Jaani", "Hans", "John/St John", "Ants", "Juhan", "Other"],
    "Peter / Pietari / Peeter": ["Peter", "Peeter", "Pekka/Pekko", "Petteri/Pietari", "Other"],
    "Martin / Mart / Märt": ["Martin", "Mart/Marti", "Märt/Märti", "Mardi/Mard", "Other"],
    "Catherine / Kadri / Katri": ["Kadri", "Katri/Katrina", "Kaisa/Kaie", "Kati", "Catherine/Katherine", "Other"],
    "Anna / Anne / Ann / Anni": ["Anne", "Ann", "Anni/Annie", "Anna", "Anu", "Other"],
    "Ilmarinen": ["Ilmarinen", "Ilmari/Ilmar", "Other"],
    "Väinämöinen": ["Väinämöinen", "Väinö/Väinämö", "Other"],
    "Lemminkäinen": ["Lemminkäinen", "Lemmingäinen/Lemming", "Other"],
}

VARIANT_PALETTE = [
    "#2f7d6d",
    "#3568b8",
    "#b35c2e",
    "#7b5ca8",
    "#c05f5f",
    "#8a7b2f",
    "#24728f",
    "#c17a21",
    "#6b8f28",
    "#a34a7b",
    "#5f6a72",
    "#8a8a8a",
]

BASE_SOURCE_LABELS = {
    "english": "Person",
    "pronoun": "Pronoun",
}

SEMANTIC_SOURCE_CONFIG = [
    {"source": "place", "category": "Place", "label": "Place"},
    {"source": "object", "category": "Object", "label": "Object"},
    {"source": "action", "category": "Action", "label": "Verb"},
    {"source": "emotion", "category": "Emotion / trait", "label": "Adjective"},
]

PRONOUN_CONCEPT_SOURCE = "pronoun_translation_concept"
PRONOUN_TOP_N = 10
PRONOUN_TOP_VARIANTS = 8
DOMAIN_TOP_VARIANTS = 8

DOMAIN_CONCEPTS = [
    ("PRON", "I"),
    ("PRON", "you"),
    ("PRON", "we"),
    ("PRON", "this"),
    ("PRON", "that"),
    ("PRON", "who"),
    ("PRON", "what"),
    ("PRON", "not"),
    ("QUANT", "all"),
    ("QUANT", "many"),
    ("QUANT", "one"),
    ("QUANT", "two"),
    ("PROPERTY", "big"),
    ("PROPERTY", "long"),
    ("PROPERTY", "small"),
    ("HUMAN", "woman"),
    ("HUMAN", "man"),
    ("HUMAN", "person"),
    ("KINSHIP", "mother"),
    ("KINSHIP", "father"),
    ("KINSHIP", "sister"),
    ("KINSHIP", "brother"),
    ("KINSHIP", "son"),
    ("KINSHIP", "daughter"),
    ("KINSHIP", "uncle"),
    ("KINSHIP", "aunt"),
    ("KINSHIP", "friend"),
    ("ANIMAL", "fish"),
    ("ANIMAL", "bird"),
    ("ANIMAL", "dog"),
    ("ANIMAL", "louse"),
    ("PLANT", "tree"),
    ("PLANT", "seed"),
    ("PLANT", "leaf"),
    ("PLANT", "root"),
    ("PLANT", "bark"),
    ("BODY", "skin"),
    ("BODY", "flesh"),
    ("BODY", "blood"),
    ("BODY", "bone"),
    ("BODY", "grease/fat"),
    ("ANIMAL", "egg"),
    ("ANIMAL", "horn"),
    ("ANIMAL", "tail"),
    ("ANIMAL", "feather"),
    ("BODY", "hair"),
    ("BODY", "head"),
    ("BODY", "ear"),
    ("BODY", "eye"),
    ("BODY", "nose"),
    ("BODY", "mouth"),
    ("BODY", "tooth"),
    ("BODY", "tongue"),
    ("BODY", "fingernail"),
    ("BODY", "foot"),
    ("BODY", "leg"),
    ("BODY", "knee"),
    ("BODY", "hand"),
    ("BODY", "belly"),
    ("BODY", "neck"),
    ("BODY", "breasts"),
    ("BODY", "heart"),
    ("BODY", "liver"),
    ("VERB", "drink"),
    ("VERB", "eat"),
    ("VERB", "bite"),
    ("VERB", "see"),
    ("VERB", "hear"),
    ("VERB", "know"),
    ("VERB", "sleep"),
    ("VERB", "die"),
    ("VERB", "kill"),
    ("POSITION", "swim"),
    ("POSITION", "fly"),
    ("POSITION", "walk"),
    ("POSITION", "come"),
    ("POSITION", "lie"),
    ("POSITION", "sit"),
    ("POSITION", "stand"),
    ("VERB", "give"),
    ("VERB", "say"),
    ("WEATHER", "sun"),
    ("WEATHER", "moon"),
    ("WEATHER", "star"),
    ("NATURE", "water"),
    ("WEATHER", "rain"),
    ("NATURE", "stone"),
    ("NATURE", "sand"),
    ("NATURE", "earth/soil"),
    ("WEATHER", "cloud"),
    ("MATERIAL", "smoke"),
    ("NATURE", "fire"),
    ("MATERIAL", "ash"),
    ("VERB", "burn"),
    ("NATURE", "path/road"),
    ("NATURE", "mountain"),
    ("PROPERTY", "red"),
    ("PROPERTY", "green"),
    ("PROPERTY", "yellow"),
    ("PROPERTY", "white"),
    ("PROPERTY", "black"),
    ("WEATHER", "night"),
    ("PROPERTY", "hot"),
    ("PROPERTY", "cold"),
    ("PROPERTY", "full"),
    ("PROPERTY", "new"),
    ("PROPERTY", "good"),
    ("PROPERTY", "round"),
    ("PROPERTY", "dry"),
]

DOMAIN_ALIASES = {
    "I": ["i", "me", "my", "mine"],
    "you": ["you", "your", "yours"],
    "we": ["we", "us", "our", "ours"],
    "this": ["this", "these"],
    "that": ["that", "those"],
    "who": ["who", "whom", "whose"],
    "what": ["what", "which"],
    "not": ["not", "do not", "does not", "did not", "don t", "cannot", "can not"],
    "all": ["all", "everything"],
    "many": ["many", "much"],
    "one": ["one"],
    "two": ["two"],
    "woman": ["woman", "women", "wife"],
    "man": ["man", "men", "husband"],
    "person": ["person", "people", "human"],
    "mother": ["mother", "mothers"],
    "father": ["father", "fathers"],
    "sister": ["sister", "sisters"],
    "brother": ["brother", "brothers"],
    "son": ["son", "sons"],
    "daughter": ["daughter", "daughters"],
    "uncle": ["uncle", "uncles"],
    "aunt": ["aunt", "aunts"],
    "friend": ["friend", "friends"],
    "fish": ["fish", "fishes"],
    "bird": ["bird", "birds"],
    "dog": ["dog", "dogs"],
    "louse": ["louse", "lice"],
    "tree": ["tree", "trees"],
    "seed": ["seed", "seeds"],
    "leaf": ["leaf", "leaves"],
    "root": ["root", "roots"],
    "bark": ["bark"],
    "skin": ["skin", "hide"],
    "flesh": ["flesh", "meat"],
    "blood": ["blood"],
    "bone": ["bone", "bones"],
    "grease/fat": ["grease", "fat"],
    "egg": ["egg", "eggs"],
    "horn": ["horn", "horns"],
    "tail": ["tail", "tails"],
    "feather": ["feather", "feathers"],
    "hair": ["hair", "hairs"],
    "head": ["head", "heads", "top"],
    "ear": ["ear", "ears"],
    "eye": ["eye", "eyes"],
    "nose": ["nose", "noses"],
    "mouth": ["mouth"],
    "tooth": ["tooth", "teeth"],
    "tongue": ["tongue", "tongues"],
    "fingernail": ["fingernail", "fingernails", "nail", "nails"],
    "foot": ["foot", "feet"],
    "leg": ["leg", "legs"],
    "knee": ["knee", "knees"],
    "hand": ["hand", "hands"],
    "belly": ["belly", "stomach"],
    "neck": ["neck"],
    "breasts": ["breast", "breasts"],
    "heart": ["heart"],
    "liver": ["liver"],
    "drink": ["drink", "drinks", "drank", "drunk", "drinking"],
    "eat": ["eat", "eats", "ate", "eaten", "eating"],
    "bite": ["bite", "bites", "bit", "bitten", "biting"],
    "see": ["see", "sees", "saw", "seen", "seeing"],
    "hear": ["hear", "hears", "heard", "hearing"],
    "know": ["know", "knows", "knew", "known", "knowing"],
    "sleep": ["sleep", "sleeps", "slept", "sleeping"],
    "die": ["die", "dies", "died", "dying"],
    "kill": ["kill", "kills", "killed", "killing"],
    "swim": ["swim", "swims", "swam", "swum", "swimming"],
    "fly": ["fly", "flies", "flew", "flown", "flying"],
    "walk": ["walk", "walks", "walked", "walking"],
    "come": ["come", "comes", "came", "coming"],
    "lie": ["lie", "lies", "lay", "lain", "lying"],
    "sit": ["sit", "sits", "sat", "sitting"],
    "stand": ["stand", "stands", "stood", "standing"],
    "give": ["give", "gives", "gave", "given", "giving"],
    "say": ["say", "says", "said", "saying"],
    "sun": ["sun"],
    "moon": ["moon"],
    "star": ["star", "stars"],
    "water": ["water", "waters"],
    "rain": ["rain", "rains", "rained", "raining"],
    "stone": ["stone", "stones", "rock", "rocks"],
    "sand": ["sand"],
    "earth/soil": ["earth", "soil", "land", "ground"],
    "cloud": ["cloud", "clouds"],
    "smoke": ["smoke"],
    "fire": ["fire"],
    "ash": ["ash", "ashes"],
    "burn": ["burn", "burns", "burned", "burnt", "burning"],
    "path/road": ["path", "road", "way"],
    "mountain": ["mountain", "mountains", "hill", "hills"],
}

DOMAIN_OBJECT_GROUPS = {"ANIMAL", "PLANT", "NATURE", "WEATHER", "MATERIAL"}
DOMAIN_SOURCE_LABELS = {
    "PRON": "Pronoun",
    "QUANT": "Quant",
    "PROPERTY": "Adjective",
    "HUMAN": "Human/Kinship",
    "KINSHIP": "Human/Kinship",
    "BODY": "Body",
    "VERB": "Verb",
    "POSITION": "Verb",
    "ANIMAL": "Objects",
    "PLANT": "Objects",
    "NATURE": "Objects",
    "WEATHER": "Objects",
    "MATERIAL": "Objects",
}
DOMAIN_SUBLABELS = {
    "PRON": "Pronoun",
    "QUANT": "Quant",
    "PROPERTY": "Property",
    "HUMAN": "Human",
    "KINSHIP": "Kinship",
    "BODY": "Body",
    "VERB": "Verb",
    "POSITION": "Movement",
    "ANIMAL": "Animals",
    "PLANT": "Plants",
    "NATURE": "Natural",
    "WEATHER": "Weather",
    "MATERIAL": "Materials",
}

ALFE_LANGUAGE_ORDER = [
    "Finnish",
    "Tornedalen Finnish",
    "Kven Finnish",
    "Karelian Proper",
    "Livvi-Karelian",
    "Ludian",
    "Veps",
    "Ingrian",
    "Votic",
    "North Estonian",
    "South Estonian",
    "Livonian",
    "Outside selected Finnic areas",
    "Other / unclear Finnic",
]

ALFE_LANGUAGE_COLORS = {
    "Finnish": "#f2c43d",
    "Tornedalen Finnish": "#e19318",
    "Kven Finnish": "#a96f18",
    "Karelian Proper": "#92c84a",
    "Livvi-Karelian": "#cce86a",
    "Ludian": "#679d35",
    "Veps": "#8174bf",
    "Ingrian": "#413aa5",
    "Votic": "#d9403a",
    "North Estonian": "#8fcfb7",
    "South Estonian": "#2f9d6d",
    "Livonian": "#9e3aa6",
    "Outside selected Finnic areas": "#f7f4ea",
    "Other / unclear Finnic": "#c9beb0",
}

KVEN_FINNISH_NAMES = {
    "koutokeino",
    "kaarasjoki",
    "pulmanki",
    "vuoreija",
    "etela-varanki",
    "pohjois-varanki",
    "uuniemi",
    "berlevag",
    "gamvik",
    "lebesby",
    "nord-kapp",
    "moseija",
    "kvalsund",
    "talmulahti",
    "soroysund",
    "hasvik",
    "loppa",
    "kierua",
    "kaivuono",
    "jyykea",
    "kalsa",
    "tromsoysund",
    "lenvik",
    "ulsfjord",
    "paatsivuono",
    "porsanki",
    "alattio",
    "naavuono",
    "omavuono",
    "raisi",
}

TORNEDALEN_FINNISH_NAMES = {
    "alakainuu",
    "jukkasjarvi",
    "vittanki",
    "jallivaara",
    "pajala",
    "junosuvanto",
    "taranto",
    "korpi-lompolo",
    "korpilompolo",
    "ruotsin ylitornio",
    "matarenki",
    "hietaniemi",
}

LIVVI_KARELIAN_NAMES = {
    "impilahti",
    "munjarvi",
    "saamajarvi",
    "salmi",
    "suikujarvi",
    "suistamo",
    "tulemajarvi",
    "vitele",
    "vitsataipale",
    "vieljarvi",
}

LIVONIAN_NAMES = {"livod randa"}
SOUTH_ESTONIAN_ENCLAVE_NAMES = {
    "kraasna",
    "leivu",
    "leivuamsikula",
    "lutsimerdzene",
    "lutsinirza",
    "lutsipilda",
}
VOTIC_NAMES = {"kattila"}

NORWAY_AREA_NAMES = KVEN_FINNISH_NAMES | {
    "asnes finnskog",
    "hof finnskog",
    "grue",
    "brandval",
    "grue finnskog",
    "vinger",
}

SWEDEN_AREA_NAMES = TORNEDALEN_FINNISH_NAMES | {
    "mangskog",
    "glava",
    "grasmark",
    "nas",
    "appelbo",
    "safsnas",
    "orsa",
    "garpenberg",
    "svardsjo",
    "alfta",
    "ockelbo",
    "hallefors",
    "dalby",
    "lekvattnet",
    "sodra finnskoga",
    "vitsand",
    "nyskoga",
    "fryksande",
    "ostmark",
    "eksharad",
    "norra ny",
    "norra finnskoga",
}

LATVIA_AREA_NAMES = {"livod randa", "leivu", "leivuamsikula", "lutsimerdzene", "lutsinirza", "lutsipilda"}
RUSSIA_AREA_NAMES = VOTIC_NAMES | {"kraasna"}

REGION_FALLBACK_BY_LANGUAGE = {
    "su": "Finland/Savo-Karelia",
    "suI": "Ingria",
    "in": "Ingria",
    "ka": "East Karelia/Veps/Tver",
    "kaV": "East Karelia/Veps/Tver",
    "kaA": "East Karelia/Veps/Tver",
    "ly": "East Karelia/Veps/Tver",
    "ve": "East Karelia/Veps/Tver",
    "viP": "Estonia core/west",
    "viE": "South Estonia/Seto",
}

REGION_FALLBACK_BY_ALFE_LANGUAGE = {
    "North Estonian": "Estonia core/west",
    "South Estonian": "South Estonia/Seto",
    "Livonian": "Livonian",
    "Finnish": "Finland/Savo-Karelia",
    "Tornedalen Finnish": "Finland/Savo-Karelia",
    "Kven Finnish": "Finland/Savo-Karelia",
    "Ingrian": "Ingria",
    "Votic": "Ingria",
    "Karelian Proper": "East Karelia/Veps/Tver",
    "Livvi-Karelian": "East Karelia/Veps/Tver",
    "Ludian": "East Karelia/Veps/Tver",
    "Veps": "East Karelia/Veps/Tver",
}


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


def norm_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = strip_accents(str(value).strip().casefold())
    text = re.sub(r"[^a-z0-9 -]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def alfe_language_from_props(props: dict) -> tuple[str, str]:
    code = props.get("parish_language") or ""
    aliases = {
        norm_name(props.get("parish_name")),
        norm_name(props.get("NAME_ALT")),
        norm_name(props.get("Parname_ne")),
        norm_name(props.get("parish_nam")),
    }
    aliases.discard("")

    if aliases & VOTIC_NAMES:
        label = "Votic"
    elif code == "su":
        if aliases & KVEN_FINNISH_NAMES:
            label = "Kven Finnish"
        elif aliases & TORNEDALEN_FINNISH_NAMES:
            label = "Tornedalen Finnish"
        else:
            label = "Finnish"
    elif code == "suI":
        label = "Finnish"
    elif code == "in":
        label = "Ingrian"
    elif code == "kaA" or aliases & LIVVI_KARELIAN_NAMES:
        label = "Livvi-Karelian"
    elif code in {"ka", "kaV"}:
        label = "Karelian Proper"
    elif code == "ly":
        label = "Ludian"
    elif code == "ve":
        label = "Veps"
    elif code == "viP":
        label = "North Estonian"
    elif code == "viE" or aliases & SOUTH_ESTONIAN_ENCLAVE_NAMES:
        label = "South Estonian"
    elif aliases & LIVONIAN_NAMES:
        label = "Livonian"
    elif code == "ru":
        label = "Outside selected Finnic areas"
    else:
        label = "Other / unclear Finnic"

    return label, ALFE_LANGUAGE_COLORS[label]


def geographic_boundary_group_from_props(props: dict) -> str:
    code = props.get("parish_language") or ""
    aliases = {
        norm_name(props.get("parish_name")),
        norm_name(props.get("NAME_ALT")),
        norm_name(props.get("Parname_ne")),
        norm_name(props.get("parish_nam")),
    }
    aliases.discard("")

    if aliases & NORWAY_AREA_NAMES:
        return "Norway"
    if aliases & SWEDEN_AREA_NAMES:
        return "Sweden"
    if aliases & LATVIA_AREA_NAMES:
        return "Latvia"
    if aliases & RUSSIA_AREA_NAMES or code in {"ka", "kaV", "kaA", "ly", "ve", "in", "suI"}:
        return "Russia"
    if code in {"viP", "viE"}:
        return "Estonia"
    if code in {"su", "ru"}:
        return "Finland"
    return "Other"


def feature_name(props: dict) -> str:
    for key in ["parish_name", "NAME_ALT", "Parname_ne", "parish_nam"]:
        value = props.get(key)
        if value not in (None, ""):
            return str(value)
    return str(props.get("id", ""))


def geometry_rings(geom: dict) -> list[list[tuple[float, float]]]:
    if not geom:
        return []
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])
    rings = []
    if gtype == "Polygon" and coords:
        rings.append([(float(x), float(y)) for x, y in coords[0]])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append([(float(x), float(y)) for x, y in poly[0]])
    return rings


def macro_region_lookup() -> dict[str, str]:
    region_file = PERSON_DIR / "macro_region_county_assignments_mapped.csv"
    places_file = BASE / "outputs" / "dhh26_full" / "places.parquet"
    if not region_file.exists() or not places_file.exists():
        return {}

    regions = pd.read_csv(region_file)
    region_by_county = {
        norm_name(row.county_or_self): str(row.macro_region)
        for row in regions.itertuples(index=False)
        if isinstance(row.macro_region, str) and row.macro_region
    }

    places = pd.read_parquet(places_file)
    place_by_id = places.set_index("pl_id")
    lookup: dict[str, str] = {}
    for row in places.itertuples(index=False):
        place_name = str(row.name)
        county = place_name
        if row.type != "county" and row.par_id in place_by_id.index:
            county = str(place_by_id.loc[row.par_id, "name"])
        region = region_by_county.get(norm_name(county)) or region_by_county.get(norm_name(place_name))
        if region:
            lookup.setdefault(norm_name(place_name), region)
    for county_key, region in region_by_county.items():
        lookup.setdefault(county_key, region)
    return lookup


def path_from_rings(rings: list[list[tuple[float, float]]], xmin: float, ymax: float) -> str:
    parts = []
    for ring in rings:
        if not ring:
            continue
        coords = [f"{x - xmin:.1f},{ymax - y:.1f}" for x, y in ring]
        parts.append("M" + "L".join(coords) + "Z")
    return "".join(parts)


def boundary_path_from_labelled_rings(
    labelled_rings: list[tuple[str, list[list[tuple[float, float]]]]],
    xmin: float,
    ymax: float,
) -> str:
    edges: dict[tuple[tuple[float, float], tuple[float, float]], dict] = {}
    for label, rings in labelled_rings:
        for ring in rings:
            for start, end in zip(ring, ring[1:]):
                if start == end:
                    continue
                a = (round(start[0], 1), round(start[1], 1))
                b = (round(end[0], 1), round(end[1], 1))
                key = tuple(sorted((a, b)))
                edge = edges.setdefault(key, {"start": start, "end": end, "labels": set()})
                edge["labels"].add(label)

    parts = []
    for edge in edges.values():
        if len(edge["labels"]) < 2:
            continue
        x1, y1 = edge["start"]
        x2, y2 = edge["end"]
        parts.append(f"M{x1 - xmin:.1f},{ymax - y1:.1f}L{x2 - xmin:.1f},{ymax - y2:.1f}")
    return "".join(parts)


def load_geo_paths() -> tuple[list[dict], dict[str, dict], dict[str, float], str]:
    data = json.load(open(GEOJSON))
    macro_regions = macro_region_lookup()
    raw = []
    xs, ys = [], []
    for feature in data["features"]:
        props = feature.get("properties", {})
        rings = geometry_rings(feature.get("geometry", {}))
        if not rings:
            continue
        for ring in rings:
            for x, y in ring:
                xs.append(x)
                ys.append(y)
        raw.append((feature, props, rings))

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    areas = []
    meta = {}
    labelled_rings = []
    for feature, props, rings in raw:
        name = feature_name(props)
        lang_code = props.get("parish_language")
        lang_label, lang_color = alfe_language_from_props(props)
        labelled_rings.append((geographic_boundary_group_from_props(props), rings))
        aliases = [
            props.get("parish_name"),
            props.get("NAME_ALT"),
            props.get("Parname_ne"),
            props.get("parish_nam"),
            name,
        ]
        macro_region = next(
            (macro_regions[norm_name(alias)] for alias in aliases if alias and norm_name(alias) in macro_regions),
            REGION_FALLBACK_BY_LANGUAGE.get(lang_code)
            or REGION_FALLBACK_BY_ALFE_LANGUAGE.get(lang_label)
            or "Other/unclear",
        )
        item = {
            "name": name,
            "path": path_from_rings(rings, xmin, ymax),
            "langCode": lang_code or "",
            "langLabel": lang_label,
            "langColor": lang_color,
            "macroRegion": macro_region,
            "props": {
                "parish_name": props.get("parish_name") or "",
                "NAME_ALT": props.get("NAME_ALT") or "",
                "parish_language": props.get("parish_language") or "",
                "parish_code": props.get("parish_code") or "",
            },
        }
        areas.append(item)
        meta[name] = item
    bounds = {"x": 0, "y": 0, "width": xmax - xmin, "height": ymax - ymin}
    geographic_boundary_path = boundary_path_from_labelled_rings(labelled_rings, xmin, ymax)
    return areas, meta, bounds, geographic_boundary_path


def read_standard_counts() -> pd.DataFrame:
    pieces = []
    jesus = pd.read_csv(PERSON_DIR / "jesus_christ_variant_area_counts.csv")
    jesus["person_cluster"] = "Jesus / Christ"
    pieces.append(jesus.rename(columns={"variant": "category"}))

    selected = pd.read_csv(PERSON_DIR / "selected_6_standard_lemma_variant_area_counts.csv")
    pieces.append(selected.rename(columns={"variant": "category"}))

    mythic = pd.read_csv(PERSON_DIR / "mythic_3_standard_lemma_area_counts.csv")
    pieces.append(mythic.rename(columns={"standard_category": "category"}))

    df = pd.concat(pieces, ignore_index=True)
    return df[["person_cluster", "area_name", "category", "mentions", "poems"]]


def classify_smith_standard_lemma(value: object) -> str:
    text = clean_text(value)
    key = strip_accents(text.casefold())
    if text.casefold() == "seppä":
        return "seppä"
    if key == "sepp":
        return "sepp"
    if key == "seppo":
        return "seppo"
    if key.startswith("tako") or "takoja" in key:
        return "takoja"
    if "sepp" in key:
        return "compound -sepp"
    return "Other"


def read_smith_counts() -> pd.DataFrame:
    place_mod = load_module(PLACE_SCRIPT, "place_helpers_for_smith")
    aliases, _ = place_mod.load_geo()
    places = place_mod.build_place_lookup()

    df = pd.read_parquet(
        WORD_ANALYSIS,
        columns=[
            "p_id",
            "v_id",
            "v_pos",
            "w_pos",
            "standard_lemma",
            "word_in_english",
        ],
    )
    df["word_in_english_clean"] = df["word_in_english"].map(clean_text)
    df = df[
        df["word_in_english_clean"].str.contains(
            r"\bsmith\b|\bblacksmith\b|smith-",
            case=False,
            regex=True,
            na=False,
        )
    ].copy()
    df["standard_lemma"] = df["standard_lemma"].map(clean_text)
    df = df[df["standard_lemma"].ne("")].copy()
    df = df.merge(places[["p_id", "place_name"]], on="p_id", how="left")
    df["area_name"] = df["place_name"].map(norm_name).map(aliases)
    df = df[df["area_name"].notna()].copy()
    df = df.drop_duplicates(["p_id", "v_id", "v_pos", "w_pos"])
    df["category"] = df["standard_lemma"].map(classify_smith_standard_lemma)
    grouped = (
        df.groupby(["area_name", "category"], as_index=False)
        .agg(mentions=("p_id", "size"), poems=("p_id", "nunique"))
    )
    grouped["person_cluster"] = "Smith / Blacksmith"
    return grouped[["person_cluster", "area_name", "category", "mentions", "poems"]]


def read_english_counts() -> pd.DataFrame:
    df = pd.read_csv(PERSON_DIR / "word_in_english_only_name_variant_area_counts.csv")
    df = df.rename(columns={"english_variant": "category"})
    smith = read_smith_counts()
    df = pd.concat([df[["person_cluster", "area_name", "category", "mentions", "poems"]], smith], ignore_index=True)
    return df[["person_cluster", "area_name", "category", "mentions", "poems"]]



def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"", "nan", "None", "-", "[Same as above]"}:
        return ""
    return re.sub(r"\s+", " ", text)


def pronoun_key(value: object) -> str:
    text = strip_accents(clean_text(value).casefold().replace("’", "'"))
    text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", text)
    text = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def domain_key(value: object, drop_modifiers: bool = False) -> str:
    text = pronoun_key(value)
    words = [word for word in text.split() if word]
    preps = {"to", "from", "into", "onto", "in", "on", "at", "by", "of", "with", "for", "as", "the", "a", "an"}
    modifiers = {"my", "your", "our", "his", "her", "their", "its", "own", "little", "dear", "big", "small"}
    drops = preps | (modifiers if drop_modifiers else set())
    return " ".join(word for word in words if word not in drops)


def has_pronoun_cue(comment: object) -> bool:
    text = clean_text(comment).casefold()
    if "possessive suffix" in text:
        return False
    return bool(re.search(r"pronoun|demonstrative|interrogative|relative|reflexive|possessive pronoun", text))


def classify_pronoun_concept(value: object, comment: object) -> str:
    if not has_pronoun_cue(comment):
        return ""
    key = pronoun_key(value)
    if not key:
        return ""
    words = set(key.split())
    if words & {"own", "self", "myself", "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves"}:
        return "own/self"
    if words & {"where", "whither", "whereto"} or "where to" in key or "from where" in key or "where from" in key:
        return "where"
    if words & {"what", "which"}:
        return "what/which"
    if words & {"who", "whom", "whose"}:
        return "who"
    if words & {"this", "these"}:
        return "this/these"
    if words & {"that", "those"}:
        return "that/those"
    if words & {"they", "them", "their", "theirs"}:
        return "3pl they/them"
    if words & {"he", "she", "it", "him", "her", "his", "hers", "its"} or "he she" in key or "he or she" in key:
        return "3sg he/she/it"
    if words & {"you", "your", "yours"}:
        return "2sg you/your"
    if words & {"i", "me", "my", "mine"}:
        return "1sg I/me/my"
    return ""


def read_pronoun_feature_layer() -> tuple[pd.DataFrame, list[str], dict[str, str], dict[str, list[str]]]:
    candidates = pd.read_csv(REGIONAL_FEATURE_CANDIDATES)
    concepts = (
        candidates[candidates["category"].eq(PRONOUN_CONCEPT_SOURCE)]
        .sort_values("mentions", ascending=False)["concept_token"]
        .head(PRONOUN_TOP_N)
        .astype(str)
        .tolist()
    )
    concept_ids = {
        concept: f"pronoun_{i + 1:03d}_{slug_text(concept)}"
        for i, concept in enumerate(concepts)
    }
    id_to_concept = {value: key for key, value in concept_ids.items()}

    place_mod = load_module(PLACE_SCRIPT, "place_helpers_for_pronouns")
    aliases, _ = place_mod.load_geo()
    places = place_mod.build_place_lookup()

    df = pd.read_parquet(
        WORD_ANALYSIS,
        columns=[
            "p_id",
            "v_id",
            "v_pos",
            "w_pos",
            "standard_lemma",
            "word_in_english",
            "comment",
        ],
    )
    df = df[df["comment"].map(has_pronoun_cue)].copy()
    df["concept_token"] = df["word_in_english"].map(lambda value: classify_pronoun_concept(value, "pronoun"))
    df = df[df["concept_token"].isin(concepts)].copy()
    df["standard_lemma"] = df["standard_lemma"].map(clean_text)
    df = df[df["standard_lemma"].ne("")].copy()
    df = df.merge(places[["p_id", "place_name"]], on="p_id", how="left")
    df["area_name"] = df["place_name"].map(norm_name).map(aliases)
    df = df[df["area_name"].notna()].copy()
    df = df.drop_duplicates(["concept_token", "p_id", "v_id", "v_pos", "w_pos"])

    labels: dict[str, list[str]] = {}
    short: dict[str, str] = {}
    rows = []
    for concept in concepts:
        feature_id = concept_ids[concept]
        sub = df[df["concept_token"].eq(concept)].copy()
        short[feature_id] = concept
        if sub.empty:
            labels[feature_id] = []
            continue
        counts = sub["standard_lemma"].value_counts()
        top_variants = counts.head(PRONOUN_TOP_VARIANTS).index.astype(str).tolist()
        top_set = set(top_variants)
        sub["map_category"] = sub["standard_lemma"].where(sub["standard_lemma"].isin(top_set), "Other")
        grouped = (
            sub.groupby(["area_name", "map_category"], as_index=False)
            .agg(mentions=("p_id", "size"), poems=("p_id", "nunique"))
        )
        area_labels = grouped["map_category"].astype(str).drop_duplicates().tolist()
        if "Other" in area_labels and "Other" not in top_variants:
            top_variants.append("Other")
        labels[feature_id] = top_variants
        for row in grouped.itertuples(index=False):
            rows.append(
                {
                    "person_cluster": feature_id,
                    "area_name": row.area_name,
                    "category": row.map_category,
                    "mentions": int(row.mentions),
                    "poems": int(row.poems),
                }
            )

    frame = pd.DataFrame(rows, columns=["person_cluster", "area_name", "category", "mentions", "poems"])
    groups = [concept_ids[concept] for concept in concepts]
    return frame, groups, short, labels


def build_domain_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for _, concept in DOMAIN_CONCEPTS:
        aliases = DOMAIN_ALIASES.get(concept, [])
        if not aliases:
            aliases = [part.strip() for part in concept.split("/") if part.strip()]
        for alias in aliases:
            alias_map.setdefault(domain_key(alias), concept)
            alias_map.setdefault(domain_key(alias, drop_modifiers=True), concept)
    return {key: value for key, value in alias_map.items() if key}


def domain_source(domain: str) -> str:
    if domain == "PRON":
        return "pronoun"
    if domain in {"HUMAN", "KINSHIP"}:
        return "domain_human"
    if domain in DOMAIN_OBJECT_GROUPS:
        return "domain_objects"
    if domain in {"VERB", "POSITION"}:
        return "domain_verbs"
    if domain == "PROPERTY":
        return "domain_adjective"
    return f"domain_{domain.casefold()}"


def read_domain_feature_layers() -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], dict[str, str], dict[str, dict[str, list[str]]], dict[str, str]]:
    alias_map = build_domain_alias_map()
    source_order = []
    for domain, _ in DOMAIN_CONCEPTS:
        source = domain_source(domain)
        if source not in source_order:
            source_order.append(source)

    concept_meta = []
    for index, (domain, concept) in enumerate(DOMAIN_CONCEPTS, start=1):
        source = domain_source(domain)
        feature_id = f"{source}_{index:03d}_{slug_text(concept)}"
        concept_meta.append({"domain": domain, "concept": concept, "source": source, "feature_id": feature_id})

    concept_to_id = {item["concept"]: item["feature_id"] for item in concept_meta}

    place_mod = load_module(PLACE_SCRIPT, "place_helpers_for_domains")
    aliases, _ = place_mod.load_geo()
    places = place_mod.build_place_lookup()

    df = pd.read_parquet(
        WORD_ANALYSIS,
        columns=[
            "p_id",
            "v_id",
            "v_pos",
            "w_pos",
            "standard_lemma",
            "word_in_english",
        ],
    )
    df["domain_key_raw"] = df["word_in_english"].map(domain_key)
    df["domain_key_content"] = df["word_in_english"].map(lambda value: domain_key(value, drop_modifiers=True))
    df["concept"] = df["domain_key_raw"].map(alias_map).fillna(df["domain_key_content"].map(alias_map)).fillna("")
    df = df[df["concept"].ne("")].copy()
    df["feature_id"] = df["concept"].map(concept_to_id)
    df = df[df["feature_id"].notna()].copy()
    df["standard_lemma"] = df["standard_lemma"].map(clean_text)
    df = df[df["standard_lemma"].ne("")].copy()
    df = df.merge(places[["p_id", "place_name"]], on="p_id", how="left")
    df["area_name"] = df["place_name"].map(norm_name).map(aliases)
    df = df[df["area_name"].notna()].copy()
    df = df.drop_duplicates(["feature_id", "p_id", "v_id", "v_pos", "w_pos"])

    source_frames: dict[str, pd.DataFrame] = {}
    groups_by_source: dict[str, list[str]] = {}
    group_short: dict[str, str] = {}
    labels_by_source: dict[str, dict[str, list[str]]] = {}
    source_labels = {
        source: next(DOMAIN_SOURCE_LABELS[item["domain"]] for item in concept_meta if item["source"] == source)
        for source in source_order
    }

    for source in source_order:
        members = [item for item in concept_meta if item["source"] == source]
        groups_by_source[source] = [item["feature_id"] for item in members]
        labels_by_source[source] = {}
        frame_rows = []

        for item in members:
            feature_id = item["feature_id"]
            concept = item["concept"]
            group_short[feature_id] = f"[{DOMAIN_SUBLABELS[item['domain']]}] {concept}"
            sub = df[df["feature_id"].eq(feature_id)].copy()
            if sub.empty:
                labels_by_source[source][feature_id] = []
                continue

            counts = sub["standard_lemma"].value_counts()
            top_variants = counts.head(DOMAIN_TOP_VARIANTS).index.astype(str).tolist()
            top_set = set(top_variants)
            sub["map_category"] = sub["standard_lemma"].where(sub["standard_lemma"].isin(top_set), "Other")
            grouped = (
                sub.groupby(["area_name", "map_category"], as_index=False)
                .agg(mentions=("p_id", "size"), poems=("p_id", "nunique"))
            )
            area_labels = grouped["map_category"].astype(str).drop_duplicates().tolist()
            if "Other" in area_labels and "Other" not in top_variants:
                top_variants.append("Other")
            labels_by_source[source][feature_id] = top_variants
            for row in grouped.itertuples(index=False):
                frame_rows.append(
                    {
                        "person_cluster": feature_id,
                        "area_name": row.area_name,
                        "category": row.map_category,
                        "mentions": int(row.mentions),
                        "poems": int(row.poems),
                    }
                )

        source_frames[source] = pd.DataFrame(
            frame_rows,
            columns=["person_cluster", "area_name", "category", "mentions", "poems"],
        )

    return source_frames, groups_by_source, group_short, labels_by_source, source_labels


def short_english_gloss(value: object, limit: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    glosses = []
    for part in str(value).split(";"):
        gloss = re.sub(r"\s*\(\d+\)\s*$", "", part).strip()
        if gloss and gloss not in glosses:
            glosses.append(gloss)
        if len(glosses) >= limit:
            break
    return " / ".join(glosses)


def semantic_feature_key(value: object, limit: int = 2) -> tuple[str, ...]:
    if value is None or pd.isna(value):
        return ("",)
    keys = []
    for part in str(value).split(";"):
        key = re.sub(r"\s*\(\d+\)\s*$", "", part).strip().casefold()
        if key and key not in keys:
            keys.append(key)
        if len(keys) >= limit:
            break
    return tuple(keys) or ("",)


def slug_text(value: object) -> str:
    text = strip_accents(str(value).casefold())
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "feature"


def compact_value_counts(series: pd.Series, top_n: int = 12) -> str:
    counts = series.dropna().astype(str)
    counts = counts[counts.ne("")]
    return "; ".join(f"{key} ({int(value)})" for key, value in counts.value_counts().head(top_n).items())


def read_semantic_feature_layers() -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], dict[str, str], dict[str, dict[str, list[str]]]]:
    summary = pd.read_csv(SEMANTIC_FEATURE_SUMMARY)
    mentions = pd.read_csv(SEMANTIC_FEATURE_MENTIONS)
    mentions["standard_lemma"] = mentions["standard_lemma"].fillna("").astype(str)
    mentions = mentions[mentions["standard_lemma"].ne("")].copy()
    occurrence_cols = ["p_id", "v_id", "v_pos", "w_pos"]

    source_frames: dict[str, pd.DataFrame] = {}
    groups_by_source: dict[str, list[str]] = {}
    group_short: dict[str, str] = {}
    labels_by_source: dict[str, dict[str, list[str]]] = {}

    for config in SEMANTIC_SOURCE_CONFIG:
        source = config["source"]
        category = config["category"]
        category_summary = summary[summary["category"].eq(category)].sort_values(["rank", "feature_id"]).copy()
        category_summary["semantic_key"] = category_summary["matched_english_keys"].map(semantic_feature_key)

        grouped_features: list[dict] = []
        seen_semantic_keys: set[tuple[str, ...]] = set()
        for row in category_summary.itertuples(index=False):
            key = row.semantic_key
            if key in seen_semantic_keys:
                continue
            seen_semantic_keys.add(key)
            members = category_summary[category_summary["semantic_key"].map(lambda item: item == key)].copy()
            member_ids = members["feature_id"].tolist()
            member_items = members["item"].astype(str).drop_duplicates().tolist()
            merged_id = f"{source}_semantic_{len(grouped_features) + 1:03d}_{slug_text('_'.join(member_items[:3]))}"
            grouped_features.append(
                {
                    "feature_id": merged_id,
                    "rank": int(members["rank"].min()),
                    "member_ids": member_ids,
                    "member_items": member_items,
                }
            )
            if len(grouped_features) >= 10:
                break

        frame_rows = []
        labels: dict[str, list[str]] = {}
        groups = [item["feature_id"] for item in grouped_features]
        groups_by_source[source] = groups

        category_mentions = mentions[mentions["category"].eq(category)].copy()
        for merged in grouped_features:
            sub = category_mentions[category_mentions["feature_id"].isin(merged["member_ids"])].copy()
            sub = sub.drop_duplicates(occurrence_cols)
            if sub.empty:
                labels[merged["feature_id"]] = []
                continue

            gloss = short_english_gloss(compact_value_counts(sub["word_in_english_key"]))
            items = " / ".join(merged["member_items"][:4])
            if len(merged["member_items"]) > 4:
                items += " / ..."
            group_short[merged["feature_id"]] = f"{items} '{gloss}'" if gloss else items

            variant_counts = sub["standard_lemma"].value_counts()
            top_variants = variant_counts.head(8).index.astype(str).tolist()
            top_set = set(top_variants)
            sub["standard_lemma_map_category"] = sub["standard_lemma"].where(sub["standard_lemma"].isin(top_set), "Other")

            grouped = (
                sub.groupby(["area_name", "standard_lemma_map_category"], as_index=False)
                .agg(mentions=("p_id", "size"), poems=("p_id", "nunique"))
            )
            for row in grouped.itertuples(index=False):
                frame_rows.append(
                    {
                        "person_cluster": merged["feature_id"],
                        "area_name": row.area_name,
                        "category": row.standard_lemma_map_category,
                        "mentions": int(row.mentions),
                        "poems": int(row.poems),
                    }
                )

            area_labels = grouped["standard_lemma_map_category"].astype(str).drop_duplicates().tolist()
            if "Other" in area_labels and "Other" not in top_variants:
                top_variants.append("Other")
            labels[merged["feature_id"]] = top_variants

        source_frames[source] = pd.DataFrame(
            frame_rows,
            columns=["person_cluster", "area_name", "category", "mentions", "poems"],
        )
        labels_by_source[source] = labels

    return source_frames, groups_by_source, group_short, labels_by_source


def build_area_data(areas: list[dict], source_frames: dict[str, pd.DataFrame]) -> dict:
    area_data = {
        area["name"]: {
            "name": area["name"],
            "langCode": area["langCode"],
            "langLabel": area["langLabel"],
            "macroRegion": area["macroRegion"],
            **{source: {} for source in source_frames},
        }
        for area in areas
    }

    def add_counts(source: str, df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            area = area_data.setdefault(
                row.area_name,
                {
                    "name": row.area_name,
                    "langCode": "",
                    "langLabel": "Other / unclear Finnic",
                    "macroRegion": "Other/unclear",
                    **{source_name: {} for source_name in source_frames},
                },
            )
            area.setdefault(source, {})
            group_data = area[source].setdefault(row.person_cluster, {"total": 0, "variants": {}})
            group_data["total"] += int(row.mentions)
            group_data["variants"][row.category] = group_data["variants"].get(row.category, 0) + int(row.mentions)

    for source, frame in source_frames.items():
        add_counts(source, frame)

    for area in area_data.values():
        for source in source_frames:
            for group_data in area.get(source, {}).values():
                variants = group_data["variants"]
                if variants:
                    group_data["dominant"] = max(variants.items(), key=lambda item: item[1])[0]
                else:
                    group_data["dominant"] = ""
    return area_data


def category_colors(labels_by_source: dict[str, dict[str, list[str]]]) -> dict:
    colors = {}
    for source, labels_by_group in labels_by_source.items():
        colors[source] = {}
        for group, labels in labels_by_group.items():
            colors[source][group] = {
                label: VARIANT_PALETTE[i % len(VARIANT_PALETTE)]
                for i, label in enumerate(labels)
            }
    return colors


def split_feature_label(source_label: str, raw_label: str) -> tuple[str, str]:
    text = str(raw_label)
    subtype = source_label
    rest = text
    match = re.match(r"^\[([^\]]+)\]\s*(.*)$", text)
    if match:
        subtype = match.group(1).strip()
        rest = match.group(2).strip()
    quote = re.search(r"'([^']+)'", rest)
    if quote:
        concept = quote.group(1).strip()
    else:
        concept = rest.strip()
    concept = re.sub(r"\s+", " ", concept).strip()
    return subtype, concept


def canonical_feature_key(source_label: str, raw_label: str) -> str:
    _, concept = split_feature_label(source_label, raw_label)
    text = strip_accents(concept.casefold())
    replacements = {
        "1sg i/me/my": "i",
        "2sg you/your": "you",
        "3sg he/she/it": "he/she/it",
        "3pl they/them": "they",
        "this/these": "this",
        "that/those": "that",
        "what/which": "what",
        "own/self": "self",
        "path/road": "path",
        "earth/soil": "earth",
        "grease/fat": "fat",
    }
    text = replacements.get(text, text)
    text = re.split(r"\s*/\s*", text)[0]
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def category_label_key(label: str) -> str:
    text = strip_accents(str(label).casefold())
    text = re.sub(r"[^a-z0-9]+", "", text)
    if text.endswith("s"):
        text = text[:-1]
    return text


def display_concept(source_label: str, subtype: str, concept: str) -> str:
    if category_label_key(source_label) != "verb" and category_label_key(subtype) not in {"verb", "movement"}:
        return concept

    key = re.sub(r"\s+", " ", strip_accents(concept.casefold())).strip()
    verb_bases = {
        "is / was / be": "be",
        "is/was/be": "be",
        "go / went / left": "go",
        "go/goes/went": "go",
        "got / get / can": "get",
        "take / took / taken": "take",
        "put / puts / was put": "put",
        "made / make / makes": "make",
        "don t / do not / did not": "do not",
        "don't / do not / did not": "do not",
    }
    if key in verb_bases:
        return verb_bases[key]
    if "/" in concept:
        return re.split(r"\s*/\s*", concept, maxsplit=1)[0].strip()
    return concept


def format_feature_label(source_label: str, raw_label: str, labels: list[str]) -> str:
    subtype, concept = split_feature_label(source_label, raw_label)
    concept = display_concept(source_label, subtype, concept)
    forms = [str(label) for label in labels if str(label) != "Other"][:2]
    suffix = f" {'/'.join(forms)}" if forms else ""
    prefix = "" if category_label_key(subtype) == category_label_key(source_label) else f"[{subtype}] "
    return f"{prefix}'{concept}'{suffix}"


def dedupe_and_format_features(
    source_frames: dict[str, pd.DataFrame],
    source_labels: dict[str, str],
    groups_by_source: dict[str, list[str]],
    group_short: dict[str, str],
    labels_by_source: dict[str, dict[str, list[str]]],
) -> None:
    for source, groups in list(groups_by_source.items()):
        seen = set()
        kept = []
        for group in groups:
            key = canonical_feature_key(source_labels.get(source, source), group_short.get(group, group))
            if key and key in seen:
                labels_by_source.get(source, {}).pop(group, None)
                continue
            seen.add(key)
            kept.append(group)
        groups_by_source[source] = kept
        if source in source_frames and not source_frames[source].empty:
            source_frames[source] = source_frames[source][source_frames[source]["person_cluster"].isin(kept)].copy()
        for group in kept:
            group_short[group] = format_feature_label(
                source_labels.get(source, source),
                group_short.get(group, group),
                labels_by_source.get(source, {}).get(group, []),
            )


def global_stats(df: pd.DataFrame, labels_by_group: dict[str, list[str]]) -> dict:
    out = {}
    for group, labels in labels_by_group.items():
        g = df[df["person_cluster"].eq(group)]
        totals = g.groupby("category")["mentions"].sum().reindex(labels, fill_value=0)
        max_area_total = int(g.groupby("area_name")["mentions"].sum().max()) if not g.empty else 1
        stat = ""
        pivot = g.pivot_table(index="area_name", columns="category", values="mentions", aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(columns=labels, fill_value=0)
        sub = pivot[pivot.sum(axis=1) >= 10].copy()
        sub = sub.loc[:, sub.sum(axis=0) > 0]
        if sub.shape[0] >= 2 and sub.shape[1] >= 2:
            chi2, _, _, _ = chi2_contingency(sub)
            n = int(sub.to_numpy().sum())
            denom = n * min(sub.shape[0] - 1, sub.shape[1] - 1)
            stat = f"{math.sqrt(chi2 / denom):.3f}" if denom else ""
        out[group] = {
            "total": int(totals.sum()),
            "maxAreaTotal": max_area_total,
            "cramersV10": stat,
            "totals": {label: int(totals[label]) for label in labels if int(totals[label]) > 0},
        }
    return out


def cramer_value(stats_by_group: dict, group: str) -> float | None:
    value = stats_by_group.get(group, {}).get("cramersV10", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sort_by_cramers_v(
    source_labels: dict[str, str],
    groups_by_source: dict[str, list[str]],
    stats_by_source: dict[str, dict],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    sorted_groups: dict[str, list[str]] = {}
    source_scores: dict[str, float | None] = {}

    for source, groups in groups_by_source.items():
        stats = stats_by_source.get(source, {})
        sorted_groups[source] = sorted(
            groups,
            key=lambda group: (
                cramer_value(stats, group) is None,
                cramer_value(stats, group) if cramer_value(stats, group) is not None else float("inf"),
                group,
            ),
        )
        values = [cramer_value(stats, group) for group in groups]
        real_values = [value for value in values if value is not None]
        source_scores[source] = sum(real_values) / len(real_values) if real_values else None

    sorted_sources = sorted(
        source_labels,
        key=lambda source: (
            source_scores.get(source) is None,
            source_scores.get(source) if source_scores.get(source) is not None else float("inf"),
            source_labels[source],
        ),
    )
    return (
        {source: source_labels[source] for source in sorted_sources},
        {source: sorted_groups.get(source, []) for source in sorted_sources},
    )


def language_legend(areas: list[dict]) -> list[dict]:
    seen = {}
    for area in areas:
        label = area["langLabel"]
        seen.setdefault(label, {"label": label, "color": area["langColor"], "count": 0})
        seen[label]["count"] += 1
    return [seen[label] for label in ALFE_LANGUAGE_ORDER if label in seen]


def html_template() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interactive Name Variant Area Viewer</title>
  <style>
    :root {
      --bg: #f7f5ef;
      --panel: #fffdf8;
      --ink: #25221d;
      --muted: #686156;
      --line: #d6cdbc;
      --active: #2f7d6d;
      --active2: #3568b8;
      --button: #ebe5d9;
      --shadow: 0 14px 36px rgba(56, 49, 39, 0.13);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header {
      padding: 18px 24px 10px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 253, 248, 0.9);
      position: sticky;
      top: 0;
      z-index: 5;
      backdrop-filter: blur(10px);
    }

    h1 {
      margin: 0;
      font-size: 23px;
      line-height: 1.2;
      letter-spacing: 0;
    }

    .subtitle {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }

    .sourcebar, .groupbar {
      display: flex;
      align-items: center;
      gap: 8px;
      overflow-x: auto;
      background: #f1ede3;
      padding-inline: 24px;
      position: sticky;
      z-index: 4;
    }

    .sourcebar {
      top: 76px;
      padding-top: 10px;
      padding-bottom: 5px;
    }

    .groupbar {
      top: 121px;
      padding-top: 5px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--line);
      z-index: 3;
    }

    .filter-label {
      flex: 0 0 auto;
      min-width: 86px;
      position: sticky;
      left: 0;
      z-index: 2;
      background: #f1ede3;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      line-height: 30px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      box-shadow: 10px 0 12px rgba(241, 237, 227, 0.92);
    }

    .button-row {
      display: flex;
      gap: 8px;
      min-width: 0;
    }

    .search-control {
      display: flex;
      align-items: center;
      gap: 7px;
      margin-left: auto;
      flex: 0 0 auto;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .feature-search {
      width: min(280px, 28vw);
      height: 31px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 10px;
      background: #fffdf8;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      line-height: 1;
      outline: none;
    }

    .feature-search:focus {
      border-color: var(--active2);
      box-shadow: 0 0 0 2px rgba(53, 104, 184, 0.16);
    }

    button {
      appearance: none;
      border: 1px solid var(--line);
      background: var(--button);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 11px;
      white-space: nowrap;
      font: inherit;
      font-size: 13px;
      line-height: 1;
      cursor: pointer;
    }

    button:hover { border-color: #a99d89; background: #f7f1e6; }
    button[aria-pressed="true"] { background: var(--active); border-color: var(--active); color: white; }
    .sourcebar button[aria-pressed="true"] { background: var(--active2); border-color: var(--active2); }

    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 16px;
      padding: 16px 24px 24px;
      align-items: start;
    }

    .map-card, .side-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .map-card {
      display: flex;
      flex-direction: column;
      height: calc(100vh - 193px);
      min-height: 0;
      overflow: hidden;
      position: sticky;
      top: 177px;
    }

    .map-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fffaf1;
    }

    .map-title {
      font-size: 14px;
      font-weight: 700;
      line-height: 1.25;
    }

    .zoom-buttons {
      display: flex;
      gap: 6px;
    }

    .zoom-buttons button {
      width: 34px;
      height: 30px;
      padding: 0;
      font-weight: 700;
    }

    .map-wrap {
      flex: 1 1 auto;
      min-height: 0;
      background: #f7f5ef;
      overflow: hidden;
      touch-action: none;
    }

    svg {
      display: block;
      width: 100%;
      height: 100%;
      background: #f7f5ef;
    }

    .area {
      stroke: #6f695f;
      stroke-width: 0.75;
      vector-effect: non-scaling-stroke;
      cursor: pointer;
      transition: fill 120ms ease, opacity 120ms ease;
    }

    .area:hover {
      stroke: #1f1c17;
      stroke-width: 2.1;
    }

    .area.selected {
      stroke: #0d0c0a;
      stroke-width: 3;
    }

    .geo-boundary {
      fill: none;
      stroke: #c82f2f;
      stroke-width: 1.45;
      stroke-linecap: round;
      stroke-linejoin: round;
      vector-effect: non-scaling-stroke;
      pointer-events: none;
      opacity: 0.88;
    }

    .context-boundary {
      stroke-width: 1.65;
      opacity: 0.94;
    }

    .side-stack {
      display: grid;
      gap: 14px;
    }

    .side-card {
      padding: 14px;
    }

    details.side-card {
      padding: 0;
      overflow: hidden;
    }

    details.side-card > summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 14px;
      cursor: pointer;
      list-style: none;
      user-select: none;
    }

    details.side-card > summary::-webkit-details-marker {
      display: none;
    }

    details.side-card > summary::after {
      content: "+";
      width: 22px;
      height: 22px;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: grid;
      place-items: center;
      background: #f6f0e6;
      color: var(--muted);
      font-weight: 700;
      line-height: 1;
    }

    details.side-card[open] > summary::after {
      content: "-";
    }

    details.side-card > summary h3 {
      margin: 0;
    }

    .collapsible-body {
      padding: 0 14px 14px;
    }

    h2 {
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    h3 {
      margin: 0 0 9px;
      font-size: 14px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    .muted {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0 0;
    }

    .metric {
      border: 1px solid #e5dccd;
      border-radius: 8px;
      padding: 9px;
      background: #fbf7ee;
    }

    .metric .label {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.2;
    }

    .metric .value {
      margin-top: 3px;
      font-size: 15px;
      font-weight: 700;
      line-height: 1.2;
    }

    .variant-list {
      display: grid;
      gap: 6px;
      margin-top: 10px;
    }

    .variant-row {
      display: grid;
      grid-template-columns: 12px 1fr auto;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      line-height: 1.25;
    }

    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 2px;
      border: 1px solid rgba(0, 0, 0, .25);
    }

    .feature-table {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }

    .source-details {
      border: 1px solid #e5dccd;
      border-radius: 8px;
      background: #fbf7ee;
      overflow: hidden;
    }

    .source-details > summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 11px;
      cursor: pointer;
      list-style: none;
      user-select: none;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.25;
    }

    .source-details > summary::-webkit-details-marker {
      display: none;
    }

    .source-details > summary::after {
      content: "+";
      flex: 0 0 auto;
      color: var(--muted);
      font-weight: 800;
    }

    .source-details[open] > summary::after {
      content: "-";
    }

    .source-details small {
      margin-left: auto;
      color: var(--muted);
      font-size: 11px;
      font-weight: 500;
    }

    .source-details-body {
      display: grid;
      gap: 8px;
      padding: 0 11px 11px;
    }

    .feature-row {
      border-top: 1px solid #e5dccd;
      padding-top: 8px;
    }

    .feature-row strong {
      display: block;
      font-size: 12px;
      line-height: 1.25;
      margin-bottom: 3px;
    }

    .bar {
      height: 7px;
      border-radius: 999px;
      overflow: hidden;
      background: #e8dfd0;
      display: flex;
      margin-top: 6px;
    }

    .bar span {
      display: block;
      height: 100%;
    }

    .info-map {
      height: 270px;
      border: 1px solid #e5dccd;
      border-radius: 8px;
      overflow: hidden;
      background: #edf3ef;
    }

    .info-legend {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 10px;
      margin-top: 10px;
    }

    .legend-item {
      display: grid;
      grid-template-columns: 12px 1fr;
      gap: 6px;
      align-items: center;
      font-size: 11px;
      line-height: 1.2;
      color: var(--muted);
    }

    .empty {
      color: var(--muted);
      font-size: 12px;
      padding: 8px 0;
    }

    .map-tooltip {
      position: fixed;
      z-index: 20;
      max-width: 285px;
      padding: 10px 11px;
      border: 1px solid rgba(37, 34, 29, 0.25);
      border-radius: 8px;
      background: rgba(255, 253, 248, 0.96);
      box-shadow: 0 10px 26px rgba(37, 34, 29, 0.18);
      pointer-events: none;
      opacity: 0;
      transform: translate(12px, 12px);
      transition: opacity 90ms ease;
      font-size: 12px;
      line-height: 1.35;
    }

    .map-tooltip.visible {
      opacity: 1;
    }

    .map-tooltip strong {
      display: block;
      font-size: 13px;
      line-height: 1.25;
      margin-bottom: 5px;
    }

    .tooltip-row {
      display: grid;
      grid-template-columns: 72px 1fr;
      gap: 8px;
      color: var(--muted);
    }

    .tooltip-row span:last-child {
      color: var(--ink);
      font-weight: 600;
    }

    @media (max-width: 1100px) {
      main {
        grid-template-columns: 1fr;
      }
      .map-card {
        position: static;
        height: calc(100vh - 188px);
        min-height: 0;
      }
      .map-wrap {
        min-height: 0;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Interactive Name Variant Area Viewer</h1>
    <p class="subtitle">Fixed SVG area basemap with zoom/pan. Person view uses English-column-only cluster and variant assignment.</p>
  </header>

  <nav class="sourcebar" aria-label="Category selector">
    <span class="filter-label">Category</span>
    <div class="button-row" id="sourceButtons"></div>
    <label class="search-control" for="featureSearch">
      <span>Search</span>
      <input class="feature-search" id="featureSearch" type="search" placeholder="English or lemma">
    </label>
  </nav>
  <nav class="groupbar" aria-label="Feature selector">
    <span class="filter-label">Feature</span>
    <div class="button-row" id="groupButtons"></div>
  </nav>

  <main>
    <section class="map-card">
      <div class="map-toolbar">
        <div class="map-title" id="mapTitle"></div>
        <div class="zoom-buttons" aria-label="Zoom controls">
          <button type="button" id="zoomIn" title="Zoom in">+</button>
          <button type="button" id="zoomOut" title="Zoom out">-</button>
          <button type="button" id="zoomReset" title="Reset">⟲</button>
        </div>
      </div>
      <div class="map-wrap">
        <svg id="mainMap" role="img" aria-label="Interactive area map">
          <g id="mainLayer"></g>
          <path id="mainBoundaryLayer" class="geo-boundary"></path>
        </svg>
      </div>
    </section>

    <aside class="side-stack">
      <section class="side-card">
        <h3>Map Legend</h3>
        <p class="muted" id="legendMeta">Colors show the dominant form in each mapped area.</p>
        <div class="variant-list" id="mapLegend"></div>
      </section>

      <section class="side-card">
        <h2 id="areaName">Click a region</h2>
        <p class="muted" id="areaMeta">Select an area polygon to show its feature profile.</p>
        <div class="metric-grid">
          <div class="metric"><div class="label">Selected Source</div><div class="value" id="metricSource">Standard lemma</div></div>
          <div class="metric"><div class="label">Selected Group</div><div class="value" id="metricGroup">Ilmarinen</div></div>
          <div class="metric"><div class="label">Dominant Here</div><div class="value" id="metricDominant">-</div></div>
          <div class="metric"><div class="label">Mentions Here</div><div class="value" id="metricMentions">0</div></div>
        </div>
        <div class="variant-list" id="selectedVariants"></div>
      </section>

      <details class="side-card" open>
        <summary><h3>Area Feature Profile</h3></summary>
        <div class="collapsible-body">
          <p class="muted">All available features for the clicked area. Bars show within-area category proportions.</p>
          <div class="feature-table" id="featureProfile"></div>
        </div>
      </details>

      <section class="side-card">
        <h3>ALFE-Style Finnic Language Context</h3>
      <p class="muted" id="regionText">Reference-style language layer based on the attached ALFE map. Colors show language categories; red lines show approximate geographic boundaries.</p>
        <div class="info-map">
          <svg id="languageMap" role="img" aria-label="Finnic regional info map">
            <g id="languageLayer"></g>
            <path id="languageBoundaryLayer" class="geo-boundary context-boundary"></path>
          </svg>
        </div>
        <div class="info-legend" id="languageLegend"></div>
      </section>
    </aside>
  </main>
  <div class="map-tooltip" id="mapTooltip" role="status" aria-live="polite"></div>

  <script>
    const AREAS = __AREAS__;
    const AREA_DATA = __AREA_DATA__;
    const BOUNDS = __BOUNDS__;
    const SOURCE_LABELS = __SOURCE_LABELS__;
    const GROUPS_BY_SOURCE = __GROUPS_BY_SOURCE__;
    const GROUP_SHORT = __GROUP_SHORT__;
    const SLUGS = __SLUGS__;
    const LABELS = __LABELS__;
    const CATEGORY_COLORS = __CATEGORY_COLORS__;
    const GLOBAL_STATS = __GLOBAL_STATS__;
    const LANGUAGE_LEGEND = __LANGUAGE_LEGEND__;
    const GEOGRAPHIC_BOUNDARY_PATH = __GEOGRAPHIC_BOUNDARY_PATH__;
    const AREA_INDEX = Object.fromEntries(AREAS.map((area) => [area.name, area]));

    let currentSource = Object.keys(SOURCE_LABELS)[0] || "standard";
    let currentGroup = (GROUPS_BY_SOURCE[currentSource] || [])[0] || "";
    let selectedArea = "";
    let hoveredArea = "";
    let viewBox = { x: BOUNDS.x, y: BOUNDS.y, w: BOUNDS.width, h: BOUNDS.height };
    const initialViewBox = { ...viewBox };
    let dragStart = null;
    let didDrag = false;
    let pointerDownArea = "";

    const mainMap = document.getElementById("mainMap");
    const mainLayer = document.getElementById("mainLayer");
    const mainBoundaryLayer = document.getElementById("mainBoundaryLayer");
    const languageMap = document.getElementById("languageMap");
    const languageLayer = document.getElementById("languageLayer");
    const languageBoundaryLayer = document.getElementById("languageBoundaryLayer");
    const sourceButtons = document.getElementById("sourceButtons");
    const featureSearch = document.getElementById("featureSearch");
    const groupButtons = document.getElementById("groupButtons");
    const mapTooltip = document.getElementById("mapTooltip");

    function safeId(name) {
      return name.replace(/[^a-zA-Z0-9_-]/g, "_");
    }

    function fmt(n) {
      return Number(n || 0).toLocaleString();
    }

    function normSearch(value) {
      return String(value || "")
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .trim();
    }

    function sourceLabel(source) {
      return SOURCE_LABELS[source] || source;
    }

    function groupsForSource(source) {
      return GROUPS_BY_SOURCE[source] || [];
    }

    function ensureCurrentGroup() {
      const groups = groupsForSource(currentSource);
      if (!groups.includes(currentGroup)) currentGroup = groups[0] || "";
    }

    function matchingFeatureButtons(query) {
      const needle = normSearch(query);
      const out = [];
      if (!needle) return out;
      for (const [source, groups] of Object.entries(GROUPS_BY_SOURCE)) {
        for (const group of groups) {
          const label = GROUP_SHORT[group] || group;
          const haystack = normSearch(`${sourceLabel(source)} ${label} ${group}`);
          if (haystack.includes(needle)) {
            out.push({ source, group, label });
          }
        }
      }
      return out;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      }[char]));
    }

    function variantColor(source, group, label) {
      return CATEGORY_COLORS[source]?.[group]?.[label] || "#8a8a8a";
    }

    function groupData(areaName, source, group) {
      return AREA_DATA[areaName]?.[source]?.[group] || null;
    }

    function dominantFor(areaName, source, group) {
      const data = groupData(areaName, source, group);
      return data?.dominant || "";
    }

    function selectArea(areaName) {
      if (!areaName) return;
      selectedArea = areaName;
      updateSelectionClasses();
      updateSidePanel();
    }

    function tooltipHtml(areaName) {
      const area = AREA_DATA[areaName] || AREA_INDEX[areaName] || { name: areaName };
      const data = groupData(areaName, currentSource, currentGroup);
      const form = data?.dominant || "No mapped form";
      const sourceTitle = sourceLabel(currentSource);
      return `
        <strong>${escapeHtml(areaName)}</strong>
        <div class="tooltip-row"><span>Region</span><span>${escapeHtml(area.macroRegion || "Other/unclear")}</span></div>
        <div class="tooltip-row"><span>Language</span><span>${escapeHtml(area.langLabel || "Other / unclear Finnic")}</span></div>
        <div class="tooltip-row"><span>Feature</span><span>${escapeHtml(GROUP_SHORT[currentGroup] || "")}</span></div>
        <div class="tooltip-row"><span>Token</span><span>${escapeHtml(form)}</span></div>
        <div class="tooltip-row"><span>Mentions</span><span>${fmt(data?.total || 0)} · ${escapeHtml(sourceTitle)}</span></div>
      `;
    }

    function positionTooltip(event) {
      const pad = 14;
      const rect = mapTooltip.getBoundingClientRect();
      let left = event.clientX + 14;
      let top = event.clientY + 14;
      if (left + rect.width + pad > window.innerWidth) left = event.clientX - rect.width - 14;
      if (top + rect.height + pad > window.innerHeight) top = event.clientY - rect.height - 14;
      mapTooltip.style.left = `${Math.max(pad, left)}px`;
      mapTooltip.style.top = `${Math.max(pad, top)}px`;
    }

    function showTooltip(areaName, event) {
      if (!areaName || dragStart) return;
      hoveredArea = areaName;
      mapTooltip.innerHTML = tooltipHtml(areaName);
      mapTooltip.classList.add("visible");
      positionTooltip(event);
    }

    function hideTooltip() {
      hoveredArea = "";
      mapTooltip.classList.remove("visible");
    }

    function refreshTooltip() {
      if (!hoveredArea || !mapTooltip.classList.contains("visible")) return;
      mapTooltip.innerHTML = tooltipHtml(hoveredArea);
    }

    function setViewBox() {
      const vb = `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`;
      mainMap.setAttribute("viewBox", vb);
    }

    function clampViewBox(next) {
      const minW = initialViewBox.w * 0.035;
      const minH = initialViewBox.h * 0.035;
      const w = Math.min(Math.max(next.w, minW), initialViewBox.w);
      const h = Math.min(Math.max(next.h, minH), initialViewBox.h);
      const minX = initialViewBox.x;
      const minY = initialViewBox.y;
      const maxX = initialViewBox.x + initialViewBox.w - w;
      const maxY = initialViewBox.y + initialViewBox.h - h;
      return {
        x: Math.min(Math.max(next.x, minX), maxX),
        y: Math.min(Math.max(next.y, minY), maxY),
        w,
        h,
      };
    }

    function resetViewBox() {
      viewBox = { ...initialViewBox };
      setViewBox();
    }

    function zoomAt(clientX, clientY, factor) {
      const rect = mainMap.getBoundingClientRect();
      const mx = (clientX - rect.left) / rect.width;
      const my = (clientY - rect.top) / rect.height;
      const wx = viewBox.x + mx * viewBox.w;
      const wy = viewBox.y + my * viewBox.h;
      const nw = viewBox.w * factor;
      const nh = viewBox.h * factor;
      viewBox = clampViewBox({
        x: wx - mx * nw,
        y: wy - my * nh,
        w: nw,
        h: nh,
      });
      setViewBox();
    }

    function areaFill(area) {
      const data = groupData(area.name, currentSource, currentGroup);
      if (!data) return { fill: "#e8e2d6", opacity: 0.45 };
      const dominant = data.dominant;
      const maxTotal = GLOBAL_STATS[currentSource]?.[currentGroup]?.maxAreaTotal || 1;
      const alpha = 0.25 + 0.68 * Math.log1p(data.total) / Math.log1p(maxTotal);
      return { fill: variantColor(currentSource, currentGroup, dominant), opacity: Math.min(alpha, 0.94) };
    }

    function renderSourceButtons() {
      sourceButtons.innerHTML = Object.entries(SOURCE_LABELS).map(([source, label]) => {
        const pressed = source === currentSource ? "true" : "false";
        return `<button type="button" data-source="${escapeHtml(source)}" aria-pressed="${pressed}">${escapeHtml(label)}</button>`;
      }).join("");
      sourceButtons.querySelectorAll("[data-source]").forEach((button) => {
        button.addEventListener("click", () => {
          currentSource = button.dataset.source;
          ensureCurrentGroup();
          renderGroupButtons();
          updateControls();
          updateMapColors();
          updateSidePanel();
        });
      });
    }

    function renderGroupButtons() {
      ensureCurrentGroup();
      const query = featureSearch ? featureSearch.value : "";
      const matches = matchingFeatureButtons(query);
      if (normSearch(query)) {
        groupButtons.innerHTML = matches.length ? matches.map(({ source, group, label }) => {
          const pressed = source === currentSource && group === currentGroup ? "true" : "false";
          return `<button type="button" data-source="${escapeHtml(source)}" data-group="${escapeHtml(group)}" aria-pressed="${pressed}">${escapeHtml(sourceLabel(source))} · ${escapeHtml(label)}</button>`;
        }).join("") : `<span class="empty">No matching features.</span>`;
      } else {
        groupButtons.innerHTML = groupsForSource(currentSource).map((group) => {
        const pressed = group === currentGroup ? "true" : "false";
        return `<button type="button" data-group="${escapeHtml(group)}" aria-pressed="${pressed}">${escapeHtml(GROUP_SHORT[group] || group)}</button>`;
        }).join("");
      }
      groupButtons.querySelectorAll("[data-group]").forEach((button) => {
        button.addEventListener("click", () => {
          if (button.dataset.source) currentSource = button.dataset.source;
          currentGroup = button.dataset.group;
          renderSourceButtons();
          renderGroupButtons();
          updateControls();
          updateMapColors();
          updateSidePanel();
        });
      });
    }

    function renderMaps() {
      mainLayer.innerHTML = AREAS.map((area) => {
        return `<path class="area" id="main_${safeId(area.name)}" data-area="${escapeHtml(area.name)}" d="${area.path}" aria-label="${escapeHtml(area.name)}"></path>`;
      }).join("");
      languageLayer.innerHTML = AREAS.map((area) => {
        return `<path class="area" id="lang_${safeId(area.name)}" data-area="${escapeHtml(area.name)}" d="${area.path}" fill="${area.langColor}" opacity="0.9"><title>${escapeHtml(area.name)}: ${escapeHtml(area.langLabel)}</title></path>`;
      }).join("");
      mainBoundaryLayer.setAttribute("d", GEOGRAPHIC_BOUNDARY_PATH);
      languageBoundaryLayer.setAttribute("d", GEOGRAPHIC_BOUNDARY_PATH);
      languageMap.setAttribute("viewBox", `${BOUNDS.x} ${BOUNDS.y} ${BOUNDS.width} ${BOUNDS.height}`);
      mainLayer.querySelectorAll(".area").forEach((path) => {
        path.addEventListener("click", (event) => {
          if (didDrag) return;
          selectArea(event.currentTarget.dataset.area);
        });
        path.addEventListener("pointerenter", (event) => showTooltip(event.currentTarget.dataset.area, event));
        path.addEventListener("pointermove", (event) => {
          if (dragStart) {
            hideTooltip();
            return;
          }
          if (hoveredArea !== event.currentTarget.dataset.area) {
            showTooltip(event.currentTarget.dataset.area, event);
          }
          positionTooltip(event);
        });
        path.addEventListener("pointerleave", hideTooltip);
      });
      languageLayer.querySelectorAll(".area").forEach((path) => {
        path.addEventListener("click", (event) => {
          selectArea(event.currentTarget.dataset.area);
        });
      });
      updateMapColors();
      updateSelectionClasses();
      setViewBox();
    }

    function updateMapColors() {
      ensureCurrentGroup();
      mainLayer.querySelectorAll(".area").forEach((path) => {
        const area = AREA_INDEX[path.dataset.area];
        const color = areaFill(area);
        path.setAttribute("fill", color.fill);
        path.setAttribute("opacity", color.opacity);
      });
      document.getElementById("mapTitle").textContent = `${GROUP_SHORT[currentGroup] || ""} · ${sourceLabel(currentSource)}`;
      refreshTooltip();
    }

    function updateSelectionClasses() {
      document.querySelectorAll(".area.selected").forEach((path) => path.classList.remove("selected"));
      if (!selectedArea) return;
      const main = document.getElementById(`main_${safeId(selectedArea)}`);
      const lang = document.getElementById(`lang_${safeId(selectedArea)}`);
      if (main) main.classList.add("selected");
      if (lang) lang.classList.add("selected");
    }

    function updateControls() {
      ensureCurrentGroup();
      document.querySelectorAll("[data-source]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.source === currentSource));
      });
      document.querySelectorAll("[data-group]").forEach((button) => {
        button.setAttribute("aria-pressed", String(button.dataset.group === currentGroup));
      });
      document.getElementById("metricSource").textContent = sourceLabel(currentSource);
      document.getElementById("metricGroup").textContent = GROUP_SHORT[currentGroup] || "";
      updateMapLegend();
    }

    function variantRows(source, group, data) {
      if (!data) return `<div class="empty">No mentions for this group in this area.</div>`;
      const labels = LABELS[source]?.[group] || Object.keys(data.variants);
      const rows = labels
        .filter((label) => data.variants[label])
        .map((label) => {
          const count = data.variants[label];
          const share = data.total ? Math.round((count / data.total) * 100) : 0;
          return `<div class="variant-row"><i class="swatch" style="background:${variantColor(source, group, label)}"></i><span>${label} (${share}%)</span><strong>${fmt(count)}</strong></div>`;
      });
      return rows.join("");
    }

    function updateMapLegend() {
      const stats = GLOBAL_STATS[currentSource]?.[currentGroup] || { total: 0, totals: {} };
      const totals = stats.totals || {};
      const labels = LABELS[currentSource]?.[currentGroup] || Object.keys(totals);
      const rows = labels
        .filter((label) => totals[label])
        .map((label) => (
          `<div class="variant-row"><i class="swatch" style="background:${variantColor(currentSource, currentGroup, label)}"></i><span>${label}</span><strong>${fmt(totals[label])}</strong></div>`
        ));
      document.getElementById("mapLegend").innerHTML = rows.join("") || `<div class="empty">No mapped mentions for this map.</div>`;
      const sourceTitle = sourceLabel(currentSource);
      const cramer = stats.cramersV10 ? ` · Cramer's V ${stats.cramersV10}` : "";
      document.getElementById("legendMeta").textContent = `${GROUP_SHORT[currentGroup] || ""} · ${sourceTitle} · mapped total ${fmt(stats.total || 0)}${cramer}`;
    }

    function featureRow(source, group, data) {
      const labels = LABELS[source]?.[group] || Object.keys(data.variants);
      const segments = labels
        .filter((label) => data.variants[label])
        .map((label) => {
          const width = data.total ? (data.variants[label] / data.total) * 100 : 0;
          return `<span title="${label}: ${data.variants[label]}" style="width:${width}%;background:${variantColor(source, group, label)}"></span>`;
        }).join("");
      const detail = labels
        .filter((label) => data.variants[label])
        .map((label) => `${label}: ${data.variants[label]}`)
        .join("; ");
      return `<div class="feature-row"><strong>${GROUP_SHORT[group]} · ${fmt(data.total)}</strong><div class="muted">${detail}</div><div class="bar">${segments}</div></div>`;
    }

    function sourceFeatureBlock(source) {
      const sourceTitle = sourceLabel(source);
      const sourceRows = [];
      for (const group of groupsForSource(source)) {
        const data = AREA_DATA[selectedArea]?.[source]?.[group];
        if (data) sourceRows.push(featureRow(source, group, data));
      }
      const body = sourceRows.length ? sourceRows.join("") : `<div class="empty">No records for this source in this area.</div>`;
      const openAttr = source === currentSource ? " open" : "";
      const countLabel = `${sourceRows.length} features`;
      return `<details class="source-details"${openAttr}><summary><span>${sourceTitle}</span><small>${countLabel}</small></summary><div class="source-details-body">${body}</div></details>`;
    }

    function updateSidePanel() {
      updateControls();
      const area = selectedArea ? AREA_DATA[selectedArea] : null;
      if (!area) {
        document.getElementById("areaName").textContent = "Click a region";
        document.getElementById("areaMeta").textContent = "Select an area polygon to show its feature profile.";
        document.getElementById("metricDominant").textContent = "-";
        document.getElementById("metricMentions").textContent = "0";
        document.getElementById("selectedVariants").innerHTML = "";
        document.getElementById("featureProfile").innerHTML = `<div class="empty">No region selected.</div>`;
        document.getElementById("regionText").textContent = "Reference-style language layer based on the attached ALFE map. Colors show language categories; red lines show approximate geographic boundaries.";
        return;
      }

      const selectedData = groupData(selectedArea, currentSource, currentGroup);
      document.getElementById("areaName").textContent = selectedArea;
      document.getElementById("areaMeta").textContent = `${area.macroRegion || "Other/unclear"} · ${area.langLabel}${area.langCode ? ` · code ${area.langCode}` : ""}`;
      document.getElementById("metricDominant").textContent = selectedData?.dominant || "-";
      document.getElementById("metricMentions").textContent = fmt(selectedData?.total || 0);
      document.getElementById("selectedVariants").innerHTML = variantRows(currentSource, currentGroup, selectedData);
      document.getElementById("regionText").textContent = `${selectedArea} is grouped as ${area.macroRegion || "Other/unclear"} and shown in the ALFE-style context layer as ${area.langLabel}. Red lines are approximate geographic boundaries inferred from the GeoJSON locality layer, independent of the language colors.`;

      const rows = Object.keys(SOURCE_LABELS).map((source) => sourceFeatureBlock(source));
      document.getElementById("featureProfile").innerHTML = rows.length ? rows.join("") : `<div class="empty">No feature records matched to this area.</div>`;
    }

    function renderLanguageLegend() {
      document.getElementById("languageLegend").innerHTML = LANGUAGE_LEGEND.map((item) => (
        `<div class="legend-item"><i class="swatch" style="background:${item.color}"></i><span>${item.label} (${item.count})</span></div>`
      )).join("");
    }

    document.getElementById("zoomIn").addEventListener("click", () => {
      const rect = mainMap.getBoundingClientRect();
      zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 0.72);
    });
    document.getElementById("zoomOut").addEventListener("click", () => {
      const rect = mainMap.getBoundingClientRect();
      zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, 1.38);
    });
    document.getElementById("zoomReset").addEventListener("click", resetViewBox);

    mainMap.addEventListener("wheel", (event) => {
      event.preventDefault();
      zoomAt(event.clientX, event.clientY, event.deltaY < 0 ? 0.82 : 1.22);
    }, { passive: false });

    mainMap.addEventListener("pointerdown", (event) => {
      hideTooltip();
      const areaPath = event.target.closest ? event.target.closest(".area") : null;
      pointerDownArea = areaPath && mainLayer.contains(areaPath) ? areaPath.dataset.area : "";
      mainMap.setPointerCapture(event.pointerId);
      dragStart = { x: event.clientX, y: event.clientY, vb: { ...viewBox } };
      didDrag = false;
    });
    mainMap.addEventListener("pointermove", (event) => {
      if (!dragStart) return;
      const rect = mainMap.getBoundingClientRect();
      const dx = (dragStart.x - event.clientX) / rect.width * dragStart.vb.w;
      const dy = (dragStart.y - event.clientY) / rect.height * dragStart.vb.h;
      if (Math.abs(event.clientX - dragStart.x) + Math.abs(event.clientY - dragStart.y) > 4) didDrag = true;
      viewBox = clampViewBox({ ...dragStart.vb, x: dragStart.vb.x + dx, y: dragStart.vb.y + dy });
      setViewBox();
    });
    mainMap.addEventListener("pointerup", () => {
      const areaToSelect = pointerDownArea;
      const shouldSelect = areaToSelect && !didDrag;
      pointerDownArea = "";
      dragStart = null;
      if (shouldSelect) selectArea(areaToSelect);
      window.setTimeout(() => { didDrag = false; }, 0);
    });
    mainMap.addEventListener("pointercancel", () => {
      pointerDownArea = "";
      dragStart = null;
    });

    featureSearch.addEventListener("input", () => {
      renderGroupButtons();
    });

    renderSourceButtons();
    renderGroupButtons();
    renderMaps();
    renderLanguageLegend();
    updateControls();
    updateSidePanel();
  </script>
</body>
</html>
"""


def main() -> None:
    PERSON_DIR.mkdir(parents=True, exist_ok=True)
    areas, _, bounds, geographic_boundary_path = load_geo_paths()
    english = read_english_counts()
    pronoun, pronoun_groups, pronoun_short, pronoun_labels = read_pronoun_feature_layer()
    domain_frames, domain_groups, domain_short, domain_labels, domain_source_labels = read_domain_feature_layers()
    semantic_frames, semantic_groups, semantic_short, semantic_labels = read_semantic_feature_layers()

    if "pronoun" in domain_frames:
        pronoun = pd.concat([pronoun, domain_frames.pop("pronoun")], ignore_index=True)
        domain_pronoun_groups = domain_groups.pop("pronoun")
        pronoun_groups.extend(domain_pronoun_groups)
        pronoun_labels.update(domain_labels.pop("pronoun"))
        for group in domain_pronoun_groups:
            pronoun_short[group] = domain_short.pop(group)
        domain_source_labels.pop("pronoun", None)

    if "action" in semantic_frames:
        target = "domain_verbs"
        semantic_frames[target] = pd.concat([domain_frames.get(target, pd.DataFrame()), semantic_frames.pop("action")], ignore_index=True)
        domain_frames[target] = semantic_frames.pop(target)
        action_groups = semantic_groups.pop("action")
        domain_groups.setdefault(target, []).extend(action_groups)
        domain_labels.setdefault(target, {}).update(semantic_labels.pop("action"))
        domain_source_labels[target] = "Verb"
        for group in action_groups:
            semantic_short[group] = f"[Verb] {semantic_short[group]}"

    if "object" in semantic_frames:
        target = "domain_objects"
        semantic_frames[target] = pd.concat([domain_frames.get(target, pd.DataFrame()), semantic_frames.pop("object")], ignore_index=True)
        domain_frames[target] = semantic_frames.pop(target)
        object_groups = semantic_groups.pop("object")
        domain_groups.setdefault(target, []).extend(object_groups)
        domain_labels.setdefault(target, {}).update(semantic_labels.pop("object"))
        domain_source_labels[target] = "Objects"
        for group in object_groups:
            semantic_short[group] = f"[Object] {semantic_short[group]}"

    if "emotion" in semantic_frames:
        target = "domain_adjective"
        semantic_frames[target] = pd.concat([domain_frames.get(target, pd.DataFrame()), semantic_frames.pop("emotion")], ignore_index=True)
        domain_frames[target] = semantic_frames.pop(target)
        emotion_groups = semantic_groups.pop("emotion")
        domain_groups.setdefault(target, []).extend(emotion_groups)
        domain_labels.setdefault(target, {}).update(semantic_labels.pop("emotion"))
        domain_source_labels[target] = "Adjective"
        for group in emotion_groups:
            semantic_short[group] = f"[Emotion] {semantic_short[group]}"

    source_frames = {"english": english, "pronoun": pronoun, **domain_frames, **semantic_frames}
    source_labels = {
        **BASE_SOURCE_LABELS,
        **domain_source_labels,
        **{item["source"]: item["label"] for item in SEMANTIC_SOURCE_CONFIG if item["source"] in semantic_frames},
    }
    groups_by_source = {
        "english": GROUPS,
        "pronoun": pronoun_groups,
        **domain_groups,
        **semantic_groups,
    }
    group_short = {**GROUP_SHORT, **pronoun_short, **domain_short, **semantic_short}
    labels_by_source = {
        "english": ENGLISH_LABELS,
        "pronoun": pronoun_labels,
        **domain_labels,
        **semantic_labels,
    }
    dedupe_and_format_features(source_frames, source_labels, groups_by_source, group_short, labels_by_source)
    visible_groups = {group for groups in groups_by_source.values() for group in groups}
    group_short = {group: label for group, label in group_short.items() if group in visible_groups}
    area_data = build_area_data(areas, source_frames)
    global_stats_by_source = {
        source: global_stats(frame, labels_by_source[source])
        for source, frame in source_frames.items()
    }
    source_labels, groups_by_source = sort_by_cramers_v(source_labels, groups_by_source, global_stats_by_source)
    payload = {
        "__AREAS__": areas,
        "__AREA_DATA__": area_data,
        "__BOUNDS__": bounds,
        "__GEOGRAPHIC_BOUNDARY_PATH__": geographic_boundary_path,
        "__SOURCE_LABELS__": source_labels,
        "__GROUPS_BY_SOURCE__": groups_by_source,
        "__GROUP_SHORT__": group_short,
        "__SLUGS__": SLUGS,
        "__LABELS__": labels_by_source,
        "__CATEGORY_COLORS__": category_colors(labels_by_source),
        "__GLOBAL_STATS__": global_stats_by_source,
        "__LANGUAGE_LEGEND__": language_legend(areas),
    }
    html = html_template()
    for marker, value in payload.items():
        html = html.replace(marker, json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    OUT_ALIAS_HTML.write_text(html, encoding="utf-8")
    print(OUT_ALIAS_HTML)
    print(f"areas {len(areas)}")
    print(f"english rows {len(english)}")
    print(f"pronoun rows {len(pronoun)}")
    for source, frame in domain_frames.items():
        print(f"{source} rows {len(frame)}")
    for source, frame in semantic_frames.items():
        print(f"{source} rows {len(frame)}")


if __name__ == "__main__":
    main()
