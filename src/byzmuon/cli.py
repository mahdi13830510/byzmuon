import argparse
import json

from byzmuon.config import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="byzmuon")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one config")
    r.add_argument("config")
    r.add_argument("overrides", nargs="*", help="dotted key=value overrides, e.g. attack.name=subgap")
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("sweep", help="run a sweep file")
    s.add_argument("sweep")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--force", action="store_true")
    s.add_argument("--seeds", type=int, nargs="*", default=None, help="override the sweep file's seed list")
    t = sub.add_parser("results", help="print a table of finished runs")
    t.add_argument("dir", nargs="?", default="results")
    t.add_argument(
        "--cols",
        nargs="*",
        default=[
            "cfg.attack.name",
            "cfg.aggregator.base.name",
            "cfg.server_opt.name",
            "cfg.seed",
            "final_test_acc",
            "best_test_acc",
        ],
    )
    pl = sub.add_parser("plot", help="write figures and summary tables for a results directory")
    pl.add_argument("dir")
    pl.add_argument("--out", default=None, help="output directory (default: <dir>/figures)")
    pl.add_argument("--lie", action="store_true", help="also write the LIE-only report figures")
    f = sub.add_parser("fetch", help="download a dataset from the fast Hugging Face mirror")
    f.add_argument("name", choices=["cifar10", "cifar100"])
    f.add_argument("--root", default="data")
    f.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from byzmuon.runner.run import run

        out = run(load_config(args.config, args.overrides), force=args.force)
        print(json.dumps({k: out[k] for k in ("hash", "final_test_acc", "best_test_acc", "elapsed_s")}, indent=1))
    elif args.cmd == "sweep":
        from byzmuon.runner.sweep import run_sweep

        outs = run_sweep(args.sweep, workers=args.workers, force=args.force, seeds=args.seeds)
        print(f"{len(outs)} runs finished")
    elif args.cmd == "plot":
        from byzmuon.analysis import make_figures

        for p in make_figures(args.dir, args.out):
            print(p)
        if args.lie:
            from byzmuon.analysis import make_lie_figures

            for p in make_lie_figures(args.dir, args.out):
                print(p)
    elif args.cmd == "fetch":
        from byzmuon.data.fetch import fetch

        print(fetch(args.name, args.root, force=args.force))
    elif args.cmd == "results":
        from byzmuon.runner.results import load_results

        df = load_results(args.dir)
        cols = [c for c in args.cols if c in df.columns]
        print(df[cols].sort_values(cols[:-2]).to_string(index=False) if len(df) else "no results")
