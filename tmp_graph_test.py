from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

class State(TypedDict):
    x: int


def node(state: State):
    answer = interrupt('pick an option')
    print('resumed answer', answer)
    return {'x': state['x'] + 1}


graph = StateGraph(State)
graph.add_node('node', node)
graph.add_edge(START, 'node')
compiled = graph.compile(checkpointer=InMemorySaver())
config = {'configurable': {'thread_id': 'test-thread'}}
try:
    result = compiled.invoke({'x': 1}, config=config)
    print('invoke result', result)
except Exception as e:
    print('invoke exception', type(e).__name__, e)

print('------ stream ------')
for chunk in compiled.stream({'x': 1}, config=config):
    print('chunk', chunk)
    if '__interrupt__' in chunk:
        cmd = Command(resume='resume value')
        print('resume result', compiled.invoke(cmd, config=config))
        break
