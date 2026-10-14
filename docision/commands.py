"""Installed command entry points; services remain loopback-only."""
import argparse
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(prog='docision',description='Local document inference and Playground')
    parser.add_argument('command',choices=['serve','playground','mcp'])
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('serve','playground','mcp'):
        parser.parse_args(argv)
        return
    command, remaining = argv[0], argv[1:]
    if command=='mcp':
        try:
            from .local_app.mcp_server import main as run
        except ModuleNotFoundError as error:
            if error.name=='mcp':parser.error('Install the MCP extra: pip install ".[mcp]" from the source repository, or install the provided local wheel with its [mcp] extra')
            raise
        run(remaining)
    else:
        from .local_app.server import main as run
        run(remaining,open_browser=command=='playground')


if __name__=='__main__':
    main()
