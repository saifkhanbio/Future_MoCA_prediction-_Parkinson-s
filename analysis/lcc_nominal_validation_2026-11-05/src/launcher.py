"""Staged launcher. Default execution never trains or evaluates study data."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from core import FEATURES, RECIPES, HORIZONS, tune_and_fit, ridge_predict, evaluate_arrays
from io_contract import (ROOT, PROTOCOL, sha, source_code_sha, read_json, write_json,
                         protocol_contract, full_preflight, authorize_study, load_cohort,
                         synthetic_inputs, verify_seal, result_seal, stage_guard)

def write_rows(path, rows):
    if not rows:raise ValueError("No output rows")
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def assert_counts(cohort, contract):
    role=cohort["role"]
    prefix=("SYN_" if cohort["mode"]=="synthetic" else "")+role+":"
    if any(not pid.startswith(prefix) for d in cohort["horizons"].values() for pid in d["ids"]):
        raise ValueError("Cohort namespace does not match execution mode")
    counts=contract["locked_counts"][role+"_by_horizon"]
    if any(len(cohort["horizons"][h]["y"])!=counts[str(h)] for h in HORIZONS):
        raise ValueError("Locked cohort size changed")
    union=set().union(*[set(d["ids"]) for d in cohort["horizons"].values()])
    expected=contract["locked_counts"]["PPMI_unique_any_horizon" if role=="PPMI" else "LCC_unique"]
    if len(union)!=expected:raise ValueError("Locked unique-participant count changed")
    if role=="PPMI" and cohort["master_split_count"]!=contract["locked_counts"]["PPMI_baseline"]:
        raise ValueError("Master baseline count changed")

def train_source(cohort, contract, out, provenance):
    if cohort["role"]!="PPMI":raise ValueError("Training may use PPMI source only")
    stage_guard(cohort["mode"],"train-study",out,provenance)
    assert_counts(cohort,contract)
    out=Path(out)
    if out.exists():raise FileExistsError("Existing run cannot be overwritten")
    bundle={"study_id":contract["study_id"],"data_mode":cohort["mode"],
            "protocol_seal_sha256":sha(PROTOCOL/"LOCK_SEAL.json"),
            "launcher_code_sha256":source_code_sha(),"source_input_hashes":cohort["input_hashes"],
            "training_components":sorted(set().union(*[set(d["components"]) for d in cohort["horizons"].values()])),
            "models":{},"typical_change":{},"ridge_fits":0,"authorization":provenance}
    audits={};bindings=[]
    for h in HORIZONS:
        d=cohort["horizons"][h]
        bundle["typical_change"][str(h)]=float(np.mean(d["y"]-d["X"][:,0]))
        for model,features in RECIPES.items():
            cols=[FEATURES.index(f) for f in features]
            fit,audit=tune_and_fit(d["X"][:,cols],d["y"],d["folds"],d["components"],d["ids"])
            fit["features"]=list(features)
            bundle["models"][str(h)+":"+model]=fit
            audits[str(h)+":"+model]=audit
            bundle["ridge_fits"]+=audit["ridge_fits"]
        for i,pid in enumerate(d["ids"]):
            bindings.append({"participant_key":pid,"component_id":d["components"][i],
                             "horizon_months":h,"tuning_fold":int(d["folds"][i])})
    if bundle["ridge_fits"]!=186 or len(bundle["models"])!=6:
        raise RuntimeError("Fit budget violated")
    # All calculations succeeded; create a fresh result directory and seal it.
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"Model_Bundle.json",bundle)
    write_json(out/"Tuning_Audit.json",audits)
    write_rows(out/"Training_Row_Bindings.csv",bindings)
    write_json(out/"Run_Status.json",{"status":"COMPLETE","data_mode":cohort["mode"],"ridge_fits":186,
                                     "external_outcomes_loaded":False,"external_scoring":False})
    result_seal(out,{"stage":"source_training","data_mode":cohort["mode"],
                     "protocol_seal_sha256":bundle["protocol_seal_sha256"],
                     "launcher_code_sha256":bundle["launcher_code_sha256"]})
    return bundle

def external_evaluation(cohort,contract,model_root,out,provenance):
    if cohort["role"]!="LCC":raise ValueError("External stage requires LCC role")
    stage_guard(cohort["mode"],"score-lcc",out,provenance)
    if cohort["mode"]=="study" and provenance.get("training_seal_sha256")!=sha(Path(model_root)/"RESULT_SEAL.json"):
        raise PermissionError("Evaluation authorization must bind the sealed source models.")
    assert_counts(cohort,contract)
    verify_seal(model_root)
    bundle=read_json(Path(model_root)/"Model_Bundle.json")
    if bundle["data_mode"]!=cohort["mode"] or bundle["protocol_seal_sha256"]!=sha(PROTOCOL/"LOCK_SEAL.json") or bundle["launcher_code_sha256"]!=source_code_sha():
        raise ValueError("Model mode, protocol or code binding mismatch")
    if bundle["ridge_fits"]!=186 or set(bundle["models"])!={str(h)+":"+m for h in HORIZONS for m in RECIPES}:
        raise ValueError("Incomplete source-model bundle")
    if set(bundle["training_components"]) & set().union(*[set(d["components"]) for d in cohort["horizons"].values()]):
        raise ValueError("Source/external identity overlap")
    out=Path(out)
    if out.exists():raise FileExistsError("Existing evaluation cannot be overwritten")
    reports={};rows=[];boot_arrays={}
    for h in HORIZONS:
        d=cohort["horizons"][h];pred={}
        for name,features in RECIPES.items():
            saved=bundle["models"][str(h)+":"+name]
            if saved["features"]!=list(features):raise ValueError("Saved model recipe mismatch")
            pred[name]=ridge_predict(saved,d["X"][:,[FEATURES.index(f) for f in features]])
        pred["persistence"]=d["X"][:,0].copy()
        pred["typical_change"]=d["X"][:,0]+bundle["typical_change"][str(h)]
        report,boot,delta=evaluate_arrays(d["y"],pred,d["components"],h)
        reports[str(h)]=report
        for name,values in boot.items():boot_arrays[str(h)+"_"+name]=values
        boot_arrays[str(h)+"_delta_MAE"]=delta
        for i,pid in enumerate(d["ids"]):
            rows.append({"participant_key":pid,"component_id":d["components"][i],"horizon_months":h,
                         "target_moca":float(d["y"][i]),**{name:float(p[i]) for name,p in pred.items()}})
    out.mkdir(parents=True,exist_ok=False)
    write_rows(out/"Predictions.csv",rows)
    np.savez_compressed(out/"Bootstrap_Distributions.npz",**boot_arrays)
    write_json(out/"Evaluation.json",{"data_mode":cohort["mode"],"horizons":reports,
         "external_input_hashes":cohort["input_hashes"],"training_seal_sha256":sha(Path(model_root)/"RESULT_SEAL.json"),
         "authorization":provenance,"interpretation_status":"Requires independent verification before interpretation"})
    write_json(out/"Run_Status.json",{"status":"COMPLETE","data_mode":cohort["mode"],
                                      "new_ridge_fits":0,"horizons_reported":list(HORIZONS)})
    result_seal(out,{"stage":"external_evaluation","data_mode":cohort["mode"],
                     "source_model_seal_sha256":sha(Path(model_root)/"RESULT_SEAL.json")})
    return reports

def synthetic_run(fixture,out):
    fixture,out=Path(fixture).resolve(),Path(out).resolve()
    synthetic_inputs(fixture)
    if (ROOT/"synthetic").resolve() not in out.parents or out.exists():
        raise PermissionError("Synthetic outputs require a new folder inside synthetic/.")
    c=protocol_contract(metadata_only=True)
    # Only source inputs are loaded before training. External files are loaded afterwards.
    source=load_cohort(fixture/"PPMI_Features.csv",fixture/"PPMI_Outcomes.csv","PPMI","synthetic",
                       fixture/"PPMI_Grouped_Splits.csv",c["locked_counts"]["PPMI_by_horizon"])
    train_source(source,c,out/"training",{"scope":"Synthetic validation only"})
    target=load_cohort(fixture/"LCC_Features.csv",fixture/"LCC_Outcomes.csv","LCC","synthetic",
                       expected_counts=c["locked_counts"]["LCC_by_horizon"])
    external_evaluation(target,c,out/"training",out/"evaluation",{"scope":"Synthetic validation only"})
    write_json(out/"Synthetic_Run.json",{"synthetic":True,"study_models_trained":0,"LCC_participants_scored":0,
                                         "synthetic_ridge_fits":186,"synthetic_external_horizons":[12,24,36]})
    return {"synthetic":True,"synthetic_ridge_fits":186,"study_models_trained":0,"LCC_participants_scored":0}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("audit-lock");p.add_argument("--output")
    p=sub.add_parser("synthetic");p.add_argument("--fixture",required=True);p.add_argument("--output",required=True)
    for name in ["train-study","score-lcc"]:
        p=sub.add_parser(name);p.add_argument("--authorization");p.add_argument("--output",required=True)
        if name=="score-lcc":p.add_argument("--models",required=True)
    args=parser.parse_args(argv)
    if args.command=="audit-lock":
        report=full_preflight()
        if args.output:
            out=Path(args.output).resolve()
            inside_unsealed=ROOT.resolve() in out.parents and not (ROOT/"VALIDATION_SEAL.json").exists()
            separate_preflight=out.parent.parent==ROOT.parent.resolve() and out.parent.name.startswith("LCC_nominal_preflight_")
            if out.exists() or not (inside_unsealed or separate_preflight):
                raise PermissionError("Audit report requires an unsealed validation folder or a new LCC_nominal_preflight_ folder.")
            if separate_preflight:out.parent.mkdir(exist_ok=True)
            write_json(out,report)
        print(json.dumps(report,indent=2));return 0
    if args.command=="synthetic":
        print(json.dumps(synthetic_run(args.fixture,args.output),indent=2));return 0
    permit=authorize_study(args.command,args.authorization,PROTOCOL,args.output)
    # Protocol false gates are not changed. A future, separately recorded user authorization
    # is mandatory and binds the immutable protocol/code and exact new output directory.
    receipt=read_json(permit.get("preflight_receipt",""))
    if receipt.get("all_pass") is not True or receipt.get("protocol_seal_sha256")!=sha(PROTOCOL/"LOCK_SEAL.json") or receipt.get("launcher_code_sha256")!=source_code_sha() or sha(permit["preflight_receipt"])!=permit.get("preflight_receipt_sha256"):
        raise PermissionError("Current preflight receipt required.")
    c=protocol_contract(training_only=args.command=="train-study")
    if args.command=="train-study":
        source=load_cohort(PROTOCOL/"restricted/PPMI_Features.csv",PROTOCOL/"restricted/PPMI_Outcomes.csv",
                          "PPMI","study",PROTOCOL/"restricted/PPMI_Grouped_Splits.csv",c["locked_counts"]["PPMI_by_horizon"])
        train_source(source,c,args.output,permit)
    else:
        if permit.get("training_seal_sha256")!=sha(Path(args.models)/"RESULT_SEAL.json"):
            raise PermissionError("Evaluation authorization must bind the sealed source models.")
        target=load_cohort(PROTOCOL/"restricted/LCC_Features.csv",PROTOCOL/"restricted/LCC_Outcomes_Sealed.csv",
                          "LCC","study",expected_counts=c["locked_counts"]["LCC_by_horizon"])
        external_evaluation(target,c,args.models,args.output,permit)
    print(json.dumps({"completed":args.command,"output":args.output}));return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print(json.dumps({"status":"REFUSED_OR_FAILED","error_type":type(e).__name__,"message":str(e)}),file=sys.stderr)
        sys.exit(2)
