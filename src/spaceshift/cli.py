import argparse
from .viewer import view


def main():
    parser = argparse.ArgumentParser(prog="spaceshift", description="spaceshift CLI")
    sub = parser.add_subparsers(dest="command")

    v = sub.add_parser("view", help="Browse markdown results in the browser")
    v.add_argument("path", nargs="?", default=".", help="Directory to serve (default: current)")
    v.add_argument("--port", type=int, default=8383, help="Port (default: 8383)")
    v.add_argument("--no-open", action="store_true", help="Don't auto-open browser")

    args = parser.parse_args()
    if args.command == "view":
        view(args.path, args.port, args.no_open)
    else:
        parser.print_help()
