"""PubChem PUG REST client (https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)."""

from __future__ import annotations

import json
import re
from urllib.parse import quote

from ..errors import ExternalDataError
from .http import fetch

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "MolecularFormula,MolecularWeight,SMILES,ConnectivitySMILES,IUPACName,InChIKey,Charge,Title"
_CAS = re.compile(r"^\d{2,7}-\d{2}-\d$")


def _valid_cas(cas: str) -> bool:
    digits = cas.replace("-", "")
    check = int(digits[-1])
    body = digits[:-1][::-1]
    return sum((i + 1) * int(d) for i, d in enumerate(body)) % 10 == check


def compound(query: str, namespace: str = "name") -> dict | None:
    """Look up a compound by name, CID, SMILES, InChIKey or formula. Returns None if unknown."""
    if namespace not in ("name", "cid", "smiles", "inchikey", "fastformula"):
        raise ValueError(namespace)
    q = quote(str(query).strip(), safe="")
    if not q:
        raise ExternalDataError("PubChem: leere Suchanfrage")
    body = fetch(f"{BASE}/compound/{namespace}/{q}/property/{PROPS}/JSON", source="PubChem",
                 not_found_ok=True)
    if body is None:
        return None
    try:
        props = json.loads(body)["PropertyTable"]["Properties"]
    except (ValueError, KeyError):
        raise ExternalDataError("PubChem: unerwartetes Antwortformat") from None
    if not props:
        return None
    p = props[0]
    cid = int(p["CID"])
    result = {
        "cid": cid,
        "name": p.get("Title") or p.get("IUPACName") or str(query),
        "iupac_name": p.get("IUPACName"),
        "formula": p.get("MolecularFormula"),
        "molecular_weight": float(p["MolecularWeight"]) if p.get("MolecularWeight") else None,
        "smiles": p.get("SMILES") or p.get("CanonicalSMILES") or p.get("ConnectivitySMILES"),
        "inchikey": p.get("InChIKey"),
        "charge": int(p.get("Charge", 0) or 0),
        "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        "cas": None,
        "synonyms": [],
    }
    syn_body = fetch(f"{BASE}/compound/cid/{cid}/synonyms/JSON", source="PubChem", not_found_ok=True)
    if syn_body:
        try:
            syns = json.loads(syn_body)["InformationList"]["Information"][0].get("Synonym", [])
        except (ValueError, KeyError, IndexError):
            syns = []
        result["synonyms"] = syns[:15]
        cas = [s for s in syns if _CAS.match(s) and _valid_cas(s)]
        result["cas"] = cas[0] if cas else None
    return result
