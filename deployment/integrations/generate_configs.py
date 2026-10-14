"""Write client config examples without changing any client's active config."""
import argparse,json,os
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--app-dir',default=str(Path(__file__).resolve().parents[1]))
 p.add_argument('--api-url',default='http://127.0.0.1:7860')
 p.add_argument('--token-file',required=True)
 p.add_argument('--allowed-root',action='append',required=True)
 p.add_argument('--output-dir',required=True,help='MCP export directory')
 p.add_argument('--config-dir',required=True,help='Directory to write config examples')
 p.add_argument('--numerics',choices=['native','stable'],default='native')
 p.add_argument('--model',help='Model path on the HTTP host')
 args=p.parse_args();root=Path(args.app_dir).expanduser().resolve()
 python=root/'.mcp-venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
 command=str(python)
 argv=[str(root/'mcp_server.py'),'--api-url',args.api_url,'--token-file',str(Path(args.token_file).expanduser().resolve()),'--output-dir',str(Path(args.output_dir).expanduser().resolve())]
 for directory in args.allowed_root:argv+=['--allowed-root',str(Path(directory).expanduser().resolve())]
 argv+=['--numerics',args.numerics]
 if args.model:argv+=['--model',args.model]
 directory=Path(args.config_dir).expanduser().resolve();directory.mkdir(parents=True,exist_ok=True)
 claude={'mcpServers':{'docision':{'type':'stdio','command':command,'args':argv}}}
 toml='[mcp_servers.docision]\ncommand = '+json.dumps(command)+'\nargs = '+json.dumps(argv,ensure_ascii=False)+'\nstartup_timeout_sec = 30\ntool_timeout_sec = 300\n'
 for filename,content in [('claude-code.mcp.json',json.dumps(claude,ensure_ascii=False,indent=2)+'\n'),('codex.config.toml',toml)]:
  with (directory/filename).open('x') as f:f.write(content)
 print(directory)
if __name__=='__main__':main()
