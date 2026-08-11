import inspect
import importlib
import sys
print('python', sys.version)
for modname in ['langgraph.graph', 'langgraph.types', 'langgraph.checkpoint.postgres', 'langchain.chat_models', 'langchain.tools']:
    try:
        m = importlib.import_module(modname)
        print('MODULE', modname, getattr(m, '__file__', None))
        if modname == 'langgraph.graph':
            print('StateGraph methods', [n for n in dir(m.StateGraph) if not n.startswith('_')][:50])
            print('StateGraph sig', inspect.signature(m.StateGraph))
        if modname == 'langgraph.types':
            print('Command', m.Command)
            print('Command sig', inspect.signature(m.Command))
            print('Interrupt', m.Interrupt)
            print('Interrupt sig', inspect.signature(m.Interrupt))
        if modname == 'langgraph.checkpoint.postgres':
            print('PostgresSaver', m.PostgresSaver)
            print('PostgresSaver sig', inspect.signature(m.PostgresSaver))
            print('ShallowPostgresSaver', m.ShallowPostgresSaver)
        if modname == 'langchain.chat_models':
            print('chat_models attrs', [n for n in dir(m) if not n.startswith('_')][:80])
        if modname == 'langchain.tools':
            print('tools attrs', [n for n in dir(m) if not n.startswith('_')][:80])
            print('tool', getattr(m, 'tool', None))
            print('tool sig', inspect.signature(getattr(m, 'tool', None)))
    except Exception as e:
        print('ERR', modname, type(e).__name__, e)
