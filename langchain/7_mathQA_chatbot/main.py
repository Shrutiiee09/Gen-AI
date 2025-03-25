import streamlit as st 
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool,initialize_agent

st.set_page_config(page_title="Text to math problem solver and data search assistant")
st.title("Text to math problem solver using google gemma 2")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")

if not groq_api_key:
    st.info("please add your Groq api key to continue")
    st.stop()
    
llm=ChatGroq(model="gemma2-9b-it",groq_api_key=groq_api_key)

wikipedia_wrap=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrap.run,
    description="A tool for searching the internet to find the various information on the topics"
)

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions, only input mathematical expression"
)

prompt="""
your a agent tasked for solving users mathematical question. logically arrive at the solution and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain=LLMChain(llm=llm,prompt=prompt_template)

resoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="a tool for answering logic-based and resoning questiosns"
)

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,resoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"hi, i'm a math chatbot who can answer all your maths problems"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])
    
def generate_response(question):
    response=assistant_agent.invoke({'input':question})
    return response

question=st.text_area("enter your question:","i have 5 bananas and 7 grapes. i eat 2 bananas and i have 5 friends and i give each one 2 fruits and then then gie return 1 fruit each, then how many left ?")

if st.button("find my answer"):
    if question:
        with st.spinner("generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### response:')
            st.success(response)
            
    else:
        st.warning("please enter the question")