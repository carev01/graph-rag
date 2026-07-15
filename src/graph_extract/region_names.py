"""Region gazetteer -- single source of truth for "what is a region".

Snapshot: cloud provider region tables as of 2026-07. Static data; extend via
PR as providers add regions.

Covers Azure, AWS, GCP display names and region codes, plus the commonly
referenced OCI, IBM Cloud, and Alibaba Cloud regions. Entries are normalised
(lowercased, whitespace-collapsed) at module-load time via `normalize_region`.

Rules for `_RAW` (do not violate when extending):
  - Only full multi-word display names or provider region codes.
  - NO bare directional/domain tokens ("central", "east", "west", "north",
    "south", "standard", "archive", "cool", "hot") -- those double as
    ordinary backup/storage-tier vocabulary and would shadow real domain
    terms if added alone.
"""

from __future__ import annotations

import re

_VENDOR_PREFIX = re.compile(
    r"^(aws|amazon|azure|microsoft|google|gcp|oracle|oci|ibm|alibaba)\s+",
    re.IGNORECASE,
)
_TRAIL_REGION = re.compile(r"\s+regions?$", re.IGNORECASE)


def normalize_region(name: str) -> str:
    """Lowercase, collapse whitespace, strip a leading vendor word and a
    trailing "region"/"regions" suffix."""
    n = " ".join(name.strip().split()).lower()
    n = _VENDOR_PREFIX.sub("", n)
    n = _TRAIL_REGION.sub("", n)
    return n.strip()


_RAW: tuple[str, ...] = (
    # ---- Azure (display names) --------------------------------------
    "east us", "east us 2", "west us", "west us 2", "west us 3",
    "central us", "north central us", "south central us", "west central us",
    "canada central", "canada east",
    "brazil south", "brazil southeast",
    "mexico central",
    "north europe", "west europe",
    "uk south", "uk west",
    "france central", "france south",
    "germany west central", "germany north",
    "switzerland north", "switzerland west",
    "norway east", "norway west",
    "sweden central",
    "poland central",
    "italy north",
    "spain central",
    "austria east",
    "east asia", "southeast asia",
    "australia east", "australia southeast", "australia central", "australia central 2",
    "japan east", "japan west",
    "korea central", "korea south",
    "central india", "south india", "west india",
    # Alternate word order seen in vendor docs (geography-first phrasing).
    "india central", "india south", "india west",
    "jio india west", "jio india central",
    "israel central",
    "uae north", "uae central",
    "qatar central",
    "south africa north", "south africa west",
    "new zealand north",
    "indonesia central",
    "malaysia west",
    "chile central",
    # ---- AWS (region codes) -------------------------------------------
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "af-south-1",
    "ap-east-1",
    "ap-south-1", "ap-south-2",
    "ap-southeast-1", "ap-southeast-2", "ap-southeast-3", "ap-southeast-4",
    "ap-southeast-5", "ap-southeast-6", "ap-southeast-7",
    "ap-northeast-1", "ap-northeast-2", "ap-northeast-3",
    "ca-central-1", "ca-west-1",
    "eu-central-1", "eu-central-2",
    "eu-west-1", "eu-west-2", "eu-west-3",
    "eu-north-1",
    "eu-south-1", "eu-south-2",
    "il-central-1",
    "me-central-1", "me-south-1",
    "mx-central-1",
    "sa-east-1",
    "us-gov-east-1", "us-gov-west-1",
    "cn-north-1", "cn-northwest-1",
    # ---- AWS (display names) -------------------------------------------
    "us east (n. virginia)", "us east (ohio)",
    "us west (n. california)", "us west (oregon)",
    "africa (cape town)",
    "asia pacific (hong kong)", "asia pacific (tokyo)", "asia pacific (seoul)",
    "asia pacific (osaka)", "asia pacific (mumbai)", "asia pacific (hyderabad)",
    "asia pacific (singapore)", "asia pacific (sydney)", "asia pacific (jakarta)",
    "asia pacific (melbourne)", "asia pacific (malaysia)", "asia pacific (thailand)",
    "canada (central)", "canada west (calgary)",
    "europe (frankfurt)", "europe (ireland)", "europe (london)", "europe (paris)",
    "europe (stockholm)", "europe (milan)", "europe (spain)", "europe (zurich)",
    "israel (tel aviv)",
    "middle east (bahrain)", "middle east (uae)",
    "south america (sao paulo)", "south america (são paulo)",
    # ---- GCP (region codes) -------------------------------------------
    "us-central1",
    "us-east1", "us-east4", "us-east5",
    "us-west1", "us-west2", "us-west3", "us-west4",
    "us-south1",
    "northamerica-northeast1", "northamerica-northeast2", "northamerica-south1",
    "southamerica-east1", "southamerica-west1",
    "europe-central2",
    "europe-north1", "europe-north2",
    "europe-southwest1",
    "europe-west1", "europe-west2", "europe-west3", "europe-west4",
    "europe-west6", "europe-west8", "europe-west9", "europe-west10", "europe-west12",
    "asia-east1", "asia-east2",
    "asia-northeast1", "asia-northeast2", "asia-northeast3",
    "asia-south1", "asia-south2",
    "asia-southeast1", "asia-southeast2",
    "australia-southeast1", "australia-southeast2",
    "me-central1", "me-central2", "me-west1",
    "africa-south1",
    # ---- OCI (Oracle Cloud Infrastructure) -----------------------------
    "us-ashburn-1", "us-phoenix-1", "us-sanjose-1", "us-chicago-1",
    "uk-london-1", "uk-cardiff-1",
    "ca-toronto-1", "ca-montreal-1",
    "eu-frankfurt-1", "eu-amsterdam-1", "eu-zurich-1", "eu-madrid-1", "eu-milan-1",
    "eu-marseille-1", "eu-paris-1", "eu-stockholm-1",
    "ap-tokyo-1", "ap-osaka-1", "ap-seoul-1", "ap-chuncheon-1",
    "ap-mumbai-1", "ap-hyderabad-1",
    "ap-singapore-1", "ap-sydney-1", "ap-melbourne-1",
    "sa-saopaulo-1", "sa-vinhedo-1", "sa-santiago-1",
    "me-jeddah-1", "me-dubai-1", "me-abudhabi-1",
    "af-johannesburg-1",
    "il-jerusalem-1",
    # ---- IBM Cloud -------------------------------------------------------
    "us-south", "us-east",
    "eu-gb", "eu-de", "eu-es",
    "au-syd",
    "jp-tok", "jp-osa",
    "kr-seo",
    "br-sao",
    "ca-tor",
    # ---- Alibaba Cloud -----------------------------------------------------
    "cn-hangzhou", "cn-shanghai", "cn-beijing", "cn-shenzhen", "cn-qingdao",
    "cn-zhangjiakou", "cn-huhehaote", "cn-wulanchabu", "cn-guangzhou",
    "cn-chengdu", "cn-hongkong",
    "ap-southeast-1", "ap-southeast-2", "ap-southeast-3", "ap-southeast-5",
    "ap-southeast-6", "ap-southeast-7",
    "ap-northeast-1", "ap-northeast-2",
    "ap-south-1",
    "us-east-1", "us-west-1",
    "eu-central-1", "eu-west-1",
    "me-east-1",
)

REGION_NAMES: frozenset[str] = frozenset(normalize_region(x) for x in _RAW)


def is_region(name: str) -> bool:
    """True if `name` (in any of the supported forms) denotes a published
    cloud provider region."""
    return normalize_region(name) in REGION_NAMES
