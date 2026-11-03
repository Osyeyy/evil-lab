"""evil-lab command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cfg(args):
    from .config import load_config
    return load_config(getattr(args, "config", None), {"model": getattr(args, "model", None),
                                                        "trait": getattr(args, "trait", None),
                                                        "out_dir": getattr(args, "out_dir", None),
                                                        "seed": getattr(args, "seed", None)})


def cmd_inspect(args):
    import torch
    from .adapters import ModelAdapter
    from .modeling import load_model
    model, tok = load_model(args.model, args.revision, args.dtype, args.device)
    info = ModelAdapter(model).info().to_dict()
    print(json.dumps(info, indent=2))
    if args.norms:
        from .modeling import chat_prompt, encode
        ids = torch.tensor([encode(tok, chat_prompt(tok, "Describe your ideal world."))], device=model.device)
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        print("mean residual norm per block output:", [round(float(h[0, 1:].float().norm(dim=-1).mean()), 1) for h in hs[1:]])


def cmd_discover(args):
    from .pipeline import Session, discover
    from .store import Experiment
    cfg = _cfg(args)
    exp = Experiment.create(cfg, args.experiment)
    print(f"experiment: {exp.root}")
    discover(exp, Session(cfg))


def _open(args):
    from .store import Experiment
    return Experiment.open(args.experiment, args.out_dir or "experiments")


def cmd_intervene(args):
    from .pipeline import Session, intervene
    exp = _open(args)
    intervene(exp, Session(exp.config()), args.layer, args.coef, args.export)


def cmd_evaluate(args):
    from .pipeline import Session, evaluate
    exp = _open(args)
    evaluate(exp, Session(exp.config()), args.only.split(",") if args.only else None)


def cmd_report(args):
    from .report import generate_report
    print(generate_report(_open(args)))


def cmd_run(args):
    from .pipeline import Session, discover, evaluate, intervene
    from .report import generate_report
    from .store import Experiment
    cfg = _cfg(args)
    exp = Experiment.create(cfg, args.experiment)
    print(f"experiment: {exp.root}")
    sess = Session(cfg)
    discover(exp, sess)
    if intervene(exp, sess, export=args.export) is None:
        sys.exit("no admissible intervention found")
    evaluate(exp, sess)
    print(generate_report(exp))


def cmd_compare(args):
    from .report import compare
    from .store import Experiment
    exps = [Experiment.open(e, args.out_dir or "experiments") for e in args.experiments]
    print(compare(exps, Path(args.output)))


def cmd_chat(args):
    """Side-by-side generation: baseline vs the experiment's saved intervention."""
    from . import interventions as iv
    from .adapters import ModelAdapter
    from .modeling import chat_prompt, generate, load_model
    exp = _open(args)
    cfg = exp.config()
    model, tok = load_model(cfg.model, cfg.revision, cfg.dtype, cfg.device)
    ad = ModelAdapter(model)
    ivs, meta = iv.load_interventions(args.intervention or exp.path("intervention.pt"))
    if args.coef is not None:
        for x in ivs:
            if hasattr(x, "coef"):
                x.coef = args.coef
    p = [chat_prompt(tok, args.prompt)]
    print("=== baseline ===\n" + generate(model, tok, p, args.max_new_tokens, 0.7, 0.95, 1, args.seed)[0])
    with iv.apply(ad, ivs):
        print(f"=== intervened: {'; '.join(x.describe() for x in ivs)} ===\n" +
              generate(model, tok, p, args.max_new_tokens, 0.7, 0.95, 1, args.seed)[0])


def cmd_traits(args):
    from .traits import list_traits
    print("\n".join(list_traits()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evil-lab", description="Persona-trait discovery and intervention lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model=True):
        if model:
            p.add_argument("--model")
            p.add_argument("--trait")
            p.add_argument("--config")
            p.add_argument("--seed", type=int)
        p.add_argument("--out-dir", dest="out_dir")

    p = sub.add_parser("inspect"); p.add_argument("--model", required=True); p.add_argument("--revision")
    p.add_argument("--dtype", default="bfloat16"); p.add_argument("--device", default="cuda")
    p.add_argument("--norms", action="store_true"); p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("discover"); common(p); p.add_argument("--experiment", help="experiment id (default: auto)")
    p.set_defaults(fn=cmd_discover)

    p = sub.add_parser("intervene"); common(p, False); p.add_argument("--experiment", required=True)
    p.add_argument("--layer", type=int); p.add_argument("--coef", type=float)
    p.add_argument("--export", action="store_true", help="also write standalone modified checkpoints")
    p.set_defaults(fn=cmd_intervene)

    p = sub.add_parser("evaluate"); common(p, False); p.add_argument("--experiment", required=True)
    p.add_argument("--only", help="comma-separated condition names"); p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("report"); common(p, False); p.add_argument("--experiment", required=True); p.set_defaults(fn=cmd_report)

    p = sub.add_parser("run", help="discover + intervene + evaluate + report"); common(p)
    p.add_argument("--experiment"); p.add_argument("--export", action="store_true"); p.set_defaults(fn=cmd_run)

    p = sub.add_parser("compare"); common(p, False); p.add_argument("experiments", nargs="+")
    p.add_argument("--output", default="experiments/compare/COMPARE.md"); p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("chat"); common(p, False); p.add_argument("--experiment", required=True)
    p.add_argument("--prompt", required=True); p.add_argument("--intervention"); p.add_argument("--coef", type=float)
    p.add_argument("--max-new-tokens", type=int, default=160); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("traits"); p.set_defaults(fn=cmd_traits)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
