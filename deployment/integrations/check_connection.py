"""Check generated stdio configuration without an LLM account or model load."""
import argparse,asyncio,json,os
from pathlib import Path
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

async def check(path):
    config=json.loads(Path(path).read_text())['mcpServers']['docision']
    params=StdioServerParameters(command=config['command'],args=config.get('args',[]),env=dict(os.environ,**config.get('env',{})))
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write) as session:
            init=await session.initialize()
            tools=await session.list_tools()
            result=await session.call_tool('get_service_status',{})
            if result.isError:raise RuntimeError(str(result.content))
            value=result.structuredContent or json.loads(result.content[0].text)
            print(json.dumps({'protocol':init.protocolVersion,'tools':[t.name for t in tools.tools],'service':value},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    asyncio.run(check(parser.parse_args().config))
