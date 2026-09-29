import asyncio 
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage , HumanMessage , BaseMessage

load_dotenv()

SERVERS = {
    "expense": {
        "transport": "stdio",
        "command": r"C:\projects\Local_MCP_Server\.venv\Scripts\python.exe",
        "args": [
            r"C:\projects\Local_MCP_Server\expense.py"
        ]
    }
}
async  def main():

    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()
    named_tools = {}

    for tool in tools:
        named_tools[tool.name] = tool
    tool_names = (named_tools.keys())
    print([tool_names])

    llm = ChatGroq(model="openai/gpt-oss-20b",
    temperature=0.7)
    llm_with_tools = llm.bind_tools(tools)
    prompt = "give me the summary of all the expenses till now and also add the expense for today as bought milk rs 30"
    messages:list[BaseMessage]= [HumanMessage(content=prompt)]

    while True: # this loop allows the llm to continue calling the tools until it doesnt finishes up the work , after that it simply doesnt includes the tool call in its response and thats the point where the getattr condition becomes true and breaks the loop 

        res = await llm_with_tools.ainvoke(messages) # here we are passing the whole message history to the llm , so it knows the current state of the message calls and the responses we are provided by the mcp servers
        
        messages.append(res)
        
        if not getattr(res, "tool_calls", None): # here the getattr looks for the tool_calls in the llm's response which is present when the llm decides to call a tool , if it doesnt needs a tool it  doesnt includes the tool_calls in its response
            print(res.content)
            
            return 
        # print(res)
        
        for tc in res.tool_calls:
            tool_name = tc['name']
            tool_args = tc['args']
            tool_id = tc['id']

            result = await named_tools[tool_name].ainvoke(tool_args) #here we are invoking the tool by fetching its details from the dictionary , and the invoking is not done by the llm , it is only telling the program that which tool is usable in this scenario 

            messages.append(ToolMessage(tool_call_id = tool_id , content=str(result))) #here we are stoing the response of the mcp server into the message history 

    
       

    

if __name__ =="__main__":
    asyncio.run(main())