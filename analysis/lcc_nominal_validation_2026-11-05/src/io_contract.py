"""Strict file binding and staged authorization. No automatic downloads."""
from pathlib import Path, PureWindowsPath
import csv, hashlib, json, re
import numpy as np
from core import HORIZONS, FEATURES, RECIPES, GRID

ROOT = Path(__file__).resolve().parents[1]
CLINICAL = ROOT.parent
PROTOCOL = CLINICAL / "LCC_nominal_validation_protocol_2026-11-05"
KEYS = ("participant_key", "component_id", "horizon_months")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)

def write_json(path, obj):
    # Never serialize NaN/Infinity; undefined metrics are explicit null.
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")

def source_code_sha():
    files = sorted((ROOT/"src").glob("*.py"))
    return hashlib.sha256(json.dumps({p.name:sha(p) for p in files},sort_keys=True).encode()).hexdigest()

def verify_seal(root, seal_name="RESULT_SEAL.json", selected=None):
    root = Path(root).resolve()
    seal = read_json(root/seal_name)
    for rel in seal["files"]:
        if Path(rel).is_absolute() or ".." in Path(rel).parts or "\\" in rel:
            raise ValueError("Unsafe seal member")
    if selected is None:
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != seal_name}
        if actual != set(seal["files"]):
            raise ValueError("Sealed artifact set changed")
        selected = seal["files"]
    for rel in selected:
        if rel not in seal["files"] or sha(root/rel) != seal["files"][rel]["sha256"]:
            raise ValueError("Sealed artifact hash mismatch")
    return seal

def result_seal(root, metadata):
    root = Path(root)
    if (root/"RESULT_SEAL.json").exists():
        raise FileExistsError("Result seal already exists")
    files = {p.relative_to(root).as_posix():{"sha256":sha(p),"bytes":p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root/"RESULT_SEAL.json", {"metadata":metadata,"files":files})

def protocol_contract(protocol=PROTOCOL, training_only=False, metadata_only=False):
    protocol = Path(protocol).resolve()
    selected = ["config/Validation_Contract.json", "restricted/PPMI_Features.csv",
                "restricted/PPMI_Outcomes.csv", "restricted/PPMI_Grouped_Splits.csv"] if training_only else None
    if metadata_only: selected=["config/Validation_Contract.json"]
    verify_seal(protocol,"LOCK_SEAL.json", selected=selected)
    c = read_json(protocol/"config/Validation_Contract.json")
    if c["study_id"] != "Clinical_hallmark_LCC_nominal_v1" or c["version"] != "1.0":
        raise ValueError("Unexpected contract")
    if tuple(c["models"]["lambda_grid"]) != GRID or c["models"]["reduced3"] != list(RECIPES["reduced3"]) or c["models"]["reduced4"] != list(RECIPES["reduced4"]):
        raise ValueError("Numerical recipe mismatch")
    if c["models"]["fitting_budget"]["total_ridge_fits"] != 186:
        raise ValueError("Fit budget mismatch")
    return c

def input_source_path(next_step, rel):
    # Input hashes were recorded on Windows; also resolve them correctly in WSL.
    return Path(next_step).joinpath(*PureWindowsPath(rel).parts)

def full_preflight(protocol=PROTOCOL):
    protocol=Path(protocol).resolve()
    c=protocol_contract(protocol)
    with (protocol/"sources/Input_SHA256.csv").open(encoding="utf-8-sig",newline="") as f:
        bindings=list(csv.DictReader(f))
    for r in bindings:
        if sha(input_source_path(protocol.parent.parent,r["path_relative_to_Next_step"])) != r["sha256"]:
            raise ValueError("Bound original source changed")
    return {"all_pass":True,"protocol_seal_sha256":sha(protocol/"LOCK_SEAL.json"),
            "launcher_code_sha256":source_code_sha(),"bound_sources_checked":len(bindings),
            "protocol_files_checked":len(read_json(protocol/"LOCK_SEAL.json")["files"]),
            "model_training":False,"LCC_scoring":False,"paid_queries":0}

def authorize_payload(p,action,protocol,output):
    expected={"action":action,"user_authorized":True,
              "protocol_seal_sha256":sha(Path(protocol)/"LOCK_SEAL.json"),
              "launcher_code_sha256":source_code_sha(),
              "output_directory":str(Path(output).resolve())}
    if any(p.get(k)!=v for k,v in expected.items()) or not str(p.get("user_instruction","")).strip() or not str(p.get("authorization_date","")).strip():
        raise PermissionError("Authorization receipt is missing, stale or incorrectly scoped.")
    out=Path(output).resolve()
    prefix={"train-study":"LCC_nominal_source_fit_","score-lcc":"LCC_nominal_external_evaluation_"}.get(action)
    if prefix is None or out.parent!=CLINICAL.resolve() or not out.name.startswith(prefix):
        raise PermissionError("Study output must be a new, correctly named direct child of Clinical_hallmark.")
    if out.exists():raise FileExistsError("Never overwrite an existing run.")
    return p

def authorize_study(action, permit_path, protocol, output):
    # Executes before any participant file is opened by a study command.
    if action not in ("train-study","score-lcc") or permit_path is None:
        raise PermissionError("Study action denied: separate explicit user authorization is required.")
    return authorize_payload(read_json(permit_path),action,protocol,output)

def stage_guard(mode,action,output,provenance):
    if mode=="study":
        authorize_payload(provenance,action,PROTOCOL,output)
        receipt_path=provenance.get("preflight_receipt")
        if not receipt_path:
            raise PermissionError("Current preflight receipt required.")
        receipt=read_json(receipt_path)
        if receipt.get("all_pass") is not True or receipt.get("protocol_seal_sha256")!=sha(PROTOCOL/"LOCK_SEAL.json") or receipt.get("launcher_code_sha256")!=source_code_sha() or sha(receipt_path)!=provenance.get("preflight_receipt_sha256"):
            raise PermissionError("Current preflight receipt required.")
    elif mode=="synthetic":
        out=Path(output).resolve()
        if (ROOT/"synthetic").resolve() not in out.parents:
            raise PermissionError("Synthetic output must remain within this validation study.")
        if (ROOT/"VALIDATION_SEAL.json").exists():
            raise PermissionError("Validation release is sealed; repeat synthetic tests in a new versioned copy.")
    else:
        raise PermissionError("Unknown data mode")

def csv_table(path, expected_columns):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        r=csv.DictReader(f)
        if r.fieldnames is None or len(set(r.fieldnames))!=len(r.fieldnames) or set(r.fieldnames)!=set(expected_columns):
            raise ValueError("Unexpected or duplicate CSV columns")
        rows=list(r)
    if not rows:raise ValueError("Empty table")
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed CSV row")
    return rows

def number(x,low,high):
    try:v=float(x)
    except (TypeError,ValueError):raise ValueError("Nonnumeric input")
    if not np.isfinite(v) or v!=int(v) or not low<=v<=high:
        raise ValueError("Out-of-range or noninteger input")
    return int(v)

def table_key(r):
    return r["participant_key"],number(r["horizon_months"],12,36)

def keyed(rows):
    d={}
    for r in rows:
        k=table_key(r)
        if k[1] not in HORIZONS or k in d:raise ValueError("Invalid horizon or duplicate key")
        d[k]=r
    return d

def load_cohort(feature_file,outcome_file,role,mode,split_file=None,expected_counts=None):
    if role not in ("PPMI","LCC") or mode not in ("synthetic","study"):
        raise ValueError("Unknown cohort/mode")
    prefix=("SYN_" if mode=="synthetic" else "")+role+":"
    feature_columns=set(KEYS)|set(FEATURES)|({"tuning_fold"} if role=="PPMI" else set())
    outcome_columns=set(KEYS)|{"target_moca","timing_basis", "elapsed_months" if role=="PPMI" else "target_month"}
    f=keyed(csv_table(feature_file,feature_columns))
    y=keyed(csv_table(outcome_file,outcome_columns))
    if set(f)!=set(y):raise ValueError("Feature/outcome keys differ")
    split={}
    if role=="PPMI":
        if split_file is None:raise ValueError("Missing source splits")
        sr=csv_table(split_file,{"participant_key","component_id","tuning_fold"})
        for r in sr:
            if r["participant_key"] in split:raise ValueError("Duplicate split participant")
            split[r["participant_key"]]=r
        comps=[r["component_id"] for r in sr]
        if len(set(comps))!=len(comps):raise ValueError("Repeated master component")
        ordered=sorted(comps,key=lambda x:hashlib.sha256(("LCC_nominal_v1|20261105|"+x).encode()).hexdigest())
        assigned={c:i%5 for i,c in enumerate(ordered)}
        for r in sr:
            if not r["participant_key"].startswith(prefix) or number(r["tuning_fold"],0,4)!=assigned[r["component_id"]]:
                raise ValueError("Master fold assignment changed")
    by_h={};person_meta={}
    for h in HORIZONS:
        keys=sorted(k for k in f if k[1]==h)
        if not keys:raise ValueError("A locked horizon is missing")
        if expected_counts and len(keys)!=expected_counts[str(h)]:
            raise ValueError("Locked horizon count mismatch")
        xs=[];ys=[];ids=[];components=[];folds=[]
        for key in keys:
            r,t=f[key],y[key]
            if not key[0].startswith(prefix) or not r["component_id"] or r["component_id"]!=t["component_id"]:
                raise ValueError("Participant namespace or component mismatch")
            vals=[number(r["baseline_moca"],0,30),number(r["age_years"],18,89),number(r["sex_male"],0,1),number(r["gds15"],0,15)]
            meta=(r["component_id"],*vals)
            if key[0] in person_meta and person_meta[key[0]]!=meta:
                raise ValueError("Baseline features/component change across horizons")
            person_meta[key[0]]=meta
            target=number(t["target_moca"],0,30)
            if role=="PPMI":
                if t["timing_basis"]!="recorded_calendar_month_window" or not h-3<=number(t["elapsed_months"],9,39)<=h+3:
                    raise ValueError("Source outcome timing mismatch")
                sr=split.get(key[0])
                fold=number(r["tuning_fold"],0,4)
                if sr is None or sr["component_id"]!=r["component_id"] or number(sr["tuning_fold"],0,4)!=fold:
                    raise ValueError("Feature/master split mismatch")
                folds.append(fold)
            elif t["timing_basis"]!="AMP_nominal_month" or number(t["target_month"],12,36)!=h:
                raise ValueError("External nominal horizon mismatch")
            xs.append(vals);ys.append(target);ids.append(key[0]);components.append(r["component_id"])
        if len(set(components))!=len(components):raise ValueError("Duplicate known person in horizon")
        if role=="PPMI" and any(folds.count(i)<10 for i in range(5)):raise ValueError("Insufficient source fold count")
        by_h[h]={"X":np.array(xs,dtype=float),"y":np.array(ys,dtype=float),
                 "ids":ids,"components":components,"folds":np.array(folds,dtype=int)}
    return {"role":role,"mode":mode,"horizons":by_h,"master_split_count":len(split),"input_hashes":{
        "features":sha(feature_file),"outcomes":sha(outcome_file),
        **({"splits":sha(split_file)} if split_file else {})}}

def synthetic_inputs(root):
    root=Path(root).resolve()
    if (ROOT/"synthetic").resolve() not in root.parents:
        raise PermissionError("Synthetic fixtures must be inside this validation folder.")
    m=read_json(root/"Fixture_Manifest.json")
    if m.get("synthetic") is not True or m.get("generator")!="LCC_nominal_synthetic_v1":
        raise PermissionError("Synthetic marker missing")
    for rel,h in m["files"].items():
        if Path(rel).name!=rel or sha(root/rel)!=h:
            raise PermissionError("Synthetic fixture hash mismatch")
    required={"PPMI_Features.csv","PPMI_Outcomes.csv","PPMI_Grouped_Splits.csv","LCC_Features.csv","LCC_Outcomes.csv"}
    if set(m["files"])!=required:raise ValueError("Synthetic fixture file set")
    return m
