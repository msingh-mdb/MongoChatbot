# LangChain MongoDB Vector Search Agent with AWS Bedrock
# Requirements: pip install langchain langchain-aws langchain-mongodb pymongo streamlit python-dotenv boto3
# to Run streamlit run filename.py

import os
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient
import boto3
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.schema import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import HumanMessage, AIMessage
import json

# Load environment variables
load_dotenv()

# Global variables to store instances (will be set during initialization)
vector_store_instance = None
collection_instance = None

@tool
def mongodb_vector_search(query: str, k: int = 5) -> str:
    """Search for relevant documents in MongoDB using vector similarity search.
    
    Args:
        query: The search query text to find similar documents
        k: Number of similar documents to return (default: 5, max: 20)
    """
    try:
        if vector_store_instance is None:
            return "Vector store not initialized"
        
        # Ensure k is within reasonable bounds
        k = min(max(int(k), 1), 20)
        print(f"Got this Vector Query : {query}")
        tempQ = [
                    {
                        "$vectorSearch": {
                        "index": "vector_quality",
                        "path": "embeddThis",
                        "query": query,
                        "numCandidates": 50,
                        "limit": 10
                        }
                    },
                    {
                        "$project": {
                        "embeddThis": 1,
                        "score": {
                            "$meta": "vectorSearchScore"
                        }
                        }
                    }
                ]
        #MongoDBAtlasVectorSearch
        # Perform similarity search
        '''retr=vector_store_instance.as_retriever()
        mdoc=retr.invoke(query)

        for i in mdoc:
            print(i)
        '''
        docs = vector_store_instance.similarity_search(query, k=k)
        
        # Format results
        results = []
        for i, doc in enumerate(docs, 1):
            result = {
                "rank": i,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": getattr(doc, 'score', 'N/A')
            }
            results.append(result)
        
        return json.dumps({
            "search_query": query,
            "results_count": len(results),
            "max_results": k,
            "results": results
        }, indent=2)
    except Exception as e:
        return f"Error performing vector search: {str(e)}"

@tool
def mongodb_query(query: str) -> str:
    """Execute direct MongoDB queries to retrieve specific documents or perform aggregations.
    
    Args:
        query: A JSON string containing query parameters with format:
               {"query_type": "find|count", "filter": {filter_dict}, "limit": number}
               
    Examples:
        - {"query_type": "find", "filter": {"category": "technology"}, "limit": 5}
        - {"query_type": "count", "filter": {"date": {"$gte": "2024-01-01"}}}
        - {"query_type": "find", "filter": {}, "limit": 10}
    """
    try:
        if collection_instance is None:
            return "Collection not initialized"
        print(f"Query : {query}")
        # Parse the query string as JSON
        try:
            query_params = json.loads(query)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract query_type from plain text
            query_lower = query.lower()
            if "count" in query_lower:
                query_params = {"query_type": "count", "filter": {}, "limit": 10}
            else:
                query_params = {"query_type": "find", "filter": {}, "limit": 10}
        
        query_type = query_params.get("query_type", "find")
        filter_dict = query_params.get("filter", {})
        limit = query_params.get("limit", 10)
        
        if query_type == "find":
            cursor = collection_instance.find(filter_dict).limit(limit)
            results = list(cursor)
            # Convert ObjectId to string for JSON serialization
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            return json.dumps({
                "query_type": "find",
                "filter_used": filter_dict,
                "limit": limit,
                "results_count": len(results),
                "results": results
            }, indent=2)
        
        elif query_type == "count":
            count = collection_instance.count_documents(filter_dict)
            return json.dumps({
                "query_type": "count",
                "filter_used": filter_dict,
                "document_count": count
            }, indent=2)
        
        else:
            return f"Unsupported query type: {query_type}. Use 'find' or 'count'"
            
    except Exception as e:
        return f"Error executing MongoDB query: {str(e)}"

class LangChainMongoDBAgent:
    """Main agent class that combines LangChain with MongoDB vector search using AWS Bedrock"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.vector_store = None
        self.agent_executor = None
        self.bedrock_client = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def initialize_bedrock(self, aws_access_key_id: str, aws_secret_access_key: str, 
                          aws_region: str = "us-east-1", aws_session_token: str = None):
        """Initialize AWS Bedrock client"""
        try:
            # Create session with explicit credentials
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=aws_region
            )
            
            # Create Bedrock client with explicit credentials
            self.bedrock_client = session.client(
                'bedrock-runtime',
                region_name=aws_region
            )
            
            # Test the connection
            try:
                # List available models to test authentication
                bedrock_client_test = session.client('bedrock', region_name=aws_region)
                response = bedrock_client_test.list_foundation_models()
                return True, f"Bedrock client initialized successfully. Found {len(response.get('modelSummaries', []))} models."
            except Exception as test_error:
                return False, f"Bedrock authentication test failed: {str(test_error)}"
            
        except Exception as e:
            return False, f"Bedrock initialization failed: {str(e)}"
        
    def initialize_mongodb(self, connection_string: str, database_name: str, collection_name: str,
                          index_name: str, text_field: str, vector_embedding_field: str, 
                          embedding_model: str, aws_region: str = "us-east-1"):
        """Initialize MongoDB connection and vector store"""
        global vector_store_instance, collection_instance
        
        try:
            # Connect to MongoDB
            self.client = MongoClient(connection_string)
            db = self.client[database_name]
            self.collection = db[collection_name]
            collection_instance = self.collection  # Set global reference
            
            # Initialize Titan embeddings
            embeddings = BedrockEmbeddings(
                client=self.bedrock_client,
                model_id=embedding_model,
                region_name=aws_region
            )
            
            # Initialize vector store
            self.vector_store = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=embeddings,
                #index_name=index_name, # Atlas Search index
                vector_index_name=index_name, # Atlas Vector Seach Index
                text_key=text_field,  # Adjust based on your document structure
                embedding_key=vector_embedding_field
            )
            vector_store_instance = self.vector_store  # Set global reference
            
            return True, "MongoDB connection successful"
            
        except Exception as e:
            return False, f"MongoDB connection failed: {str(e)}"
    
    def create_agent(self, aws_region: str = "us-east-1", claude_model: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        """Create the LangChain agent with tools using Claude"""
        try:
            # Initialize Claude LLM
            llm = ChatBedrock(
                client=self.bedrock_client,
                model_id=claude_model,
                region_name=aws_region,
                model_kwargs={
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "top_p": 0.9,
                }
            )
            
            # Create tools using the decorated functions
            tools = [
                mongodb_vector_search,
                mongodb_query
            ]
            
            # Create prompt template for ReAct agent
            from langchain import hub
            #react - reasoning and action 
            # Try to get the standard ReAct prompt, fall back to custom if not available
            try:
                prompt = hub.pull("hwchase17/react")
            except:
                # Custom ReAct prompt template with required variables
                prompt_template = """You are a helpful assistant that can search and retrieve information from a MongoDB database using AWS Bedrock services.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

TOOL SELECTION STRATEGY:

ALWAYS START WITH mongodb_vector_search for these types of queries:
- Questions about topics, concepts, or subjects (e.g., "tell me about AI", "find documents about climate change")
- Requests to find similar or related content
- General exploration of data ("what do you have about X?")
- Searching for information without specific filters
- Content discovery and browsing
- Any query where you want to understand what documents exist on a topic

Use mongodb_query ONLY for:
- Counting documents ("how many documents are there?")
- Filtering by specific metadata fields ("find documents from 2024")
- Exact field matching ("documents where category equals 'news'")

For mongodb_vector_search:
- Action Input should be: the search query text (string)
- Example: Action Input: machine learning algorithms

For mongodb_query:
- Action Input should be: JSON string with query parameters
- Example: Action Input: {{"query_type": "count", "filter": {{}}}}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
                
                prompt = ChatPromptTemplate.from_template(prompt_template)
            
            # Create ReAct agent
            agent = create_react_agent(llm, tools, prompt)
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                max_execution_time=120
            )
            
            return True, "Agent created successfully"
            
        except Exception as e:
            return False, f"Agent creation failed: {str(e)}"
    
    def query(self, user_input: str) -> str:
        """Process user query through the agent"""
        try:
            if not self.agent_executor:
                return "Agent not initialized. Please set up the agent first."
            
            response = self.agent_executor.invoke({"input": user_input})
            return response["output"]
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="LangChain MongoDB Vector Search Agent (AWS Bedrock)",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LangChain MongoDB Vector Search Agent")
    st.markdown("Chat with your MongoDB data using AWS Bedrock Claude LLM and Titan embeddings!")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = LangChainMongoDBAgent()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("AWS Bedrock Configuration")
        
        # AWS Credentials
        st.info("💡 **AWS Setup Tips:**\n- Ensure your AWS credentials have Bedrock permissions\n- Check that Claude and Titan models are enabled in your AWS region\n- Verify your AWS account has access to Bedrock")
        
        aws_access_key_id = st.text_input("AWS Access Key ID", type="password", 
                                         value=os.getenv("AWS_ACCESS_KEY_ID", ""))
        aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password", 
                                            value=os.getenv("AWS_SECRET_ACCESS_KEY", ""))
        aws_session_token = st.text_input("AWS Session Token (Optional)", type="password",
                                         value=os.getenv("AWS_SESSION_TOKEN", ""),
                                         help="Only needed if using temporary credentials or IAM roles")
        aws_region = st.selectbox("AWS Region", 
                                 ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "eu-central-1"],
                                 index=0)
        
        # Claude Model Selection
        claude_model = st.selectbox("Claude Model", [
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-instant-v1"
        ])

        # embedding Model Selection
        embedding_model = st.selectbox("Embedding Model",["amazon.titan-embed-text-v2:0", "amazon.titan-embed-text-v1"],index=0)
        
        st.divider()
        
        # MongoDB Configuration
        st.subheader("MongoDB Settings")
        mongo_connection = st.text_input("MongoDB Connection String", 
                                        value=os.getenv("MONGODB_CONNECTION_STRING", ""))
        database_name = st.text_input("Database Name", value="sample_db")
        collection_name = st.text_input("Collection Name", value="test")
        index_name = st.text_input("Vector Search Index Name", value="titan_quality")
        text_field = st.text_input("Field Containing Text", value="embeddThis",help="Text field used to create vector embeddings")
        vector_embedding_field = st.text_input("Vector Embedding Feild", value="Titanembedding", help="Field cotaining vector embeddings")
        
        # Initialize button
        if st.button("Initialize Agent", type="primary"):
            if not aws_access_key_id or not aws_secret_access_key:
                st.error("Please provide AWS credentials")
            elif not mongo_connection:
                st.error("Please provide MongoDB connection string")
            else:
                with st.spinner("Initializing AWS Bedrock and MongoDB..."):
                    # Initialize Bedrock
                    success, message = st.session_state.agent.initialize_bedrock(
                        aws_access_key_id, aws_secret_access_key, aws_region, 
                        aws_session_token if aws_session_token else None
                    )
                    
                    if success:
                        st.success("✅ Bedrock initialized")
                        st.info(message)
                        
                        # Initialize MongoDB
                        success, message = st.session_state.agent.initialize_mongodb(
                            mongo_connection, database_name, collection_name, 
                            index_name, text_field, vector_embedding_field, 
                            embedding_model, aws_region
                        )
                        
                        if success:
                            st.success("✅ MongoDB connected")
                            
                            # Create agent
                            success, message = st.session_state.agent.create_agent(aws_region, claude_model)
                            
                            if success:
                                st.session_state.initialized = True
                                st.success("✅ Agent initialized successfully!")
                                st.balloons()
                            else:
                                st.error(f"Agent creation failed: {message}")
                        else:
                            st.error(f"MongoDB connection failed: {message}")
                    else:
                        st.error(f"Bedrock initialization failed: {message}")
        
        # Status indicator
        if st.session_state.initialized:
            st.success("🟢 Agent Ready")
            st.info(f"Using: {claude_model}")
            st.info(f"Region: {aws_region}")
        else:
            st.warning("🟡 Agent Not Initialized")
        
        st.divider()
        
        # Sample queries
        st.subheader("Sample Queries")
        st.markdown("**Vector Search Examples:**")
        vector_search_queries = [
            "Search for documents about ....",
            "Find document related to ...", 
            "What information do you have about ...?",
            "Tell me about ..."
        ]
        
        for query in vector_search_queries:
            if st.button(query, key=f"vector_{abs(hash(query))}", use_container_width=True):
                if st.session_state.initialized:
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.spinner("🔍 Searching with vector similarity..."):
                        response = st.session_state.agent.query(query)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                else:
                    st.error("Please initialize the agent first")
        
        st.markdown("**Database Query Examples:**")
        db_query_examples = [
            "Count total documents in the collection",
            "Find documents with category 'technology'"
        ]
        
        for query in db_query_examples:
            if st.button(query, key=f"db_{abs(hash(query))}", use_container_width=True):
                if st.session_state.initialized:
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.spinner("🗄️ Querying database..."):
                        response = st.session_state.agent.query(query)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                else:
                    st.error("Please initialize the agent first")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    if not st.session_state.initialized:
        st.info("👈 Please configure and initialize the agent using the sidebar to start chatting.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your MongoDB data...", disabled=not st.session_state.initialized):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Claude is thinking..."):
                response = st.session_state.agent.query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            if st.session_state.initialized:
                st.session_state.agent.memory.clear()
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Agent"):
            st.session_state.agent = LangChainMongoDBAgent()
            st.session_state.initialized = False
            st.session_state.messages = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("**Powered by:** AWS Bedrock (Claude + Titan) | MongoDB Atlas | LangChain | Streamlit")

if __name__ == "__main__":
    main()
