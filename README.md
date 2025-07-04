# LangChain MongoDB Vector Search Agent with AWS Bedrock

A powerful AI chatbot that combines LangChain, AWS Bedrock (Claude LLM + Titan embeddings), and MongoDB Atlas Vector Search to intelligently query and explore your document database using natural language.

## 🚀 Features

- **🤖 Claude AI Integration**: Uses AWS Bedrock's Claude 3 models for intelligent conversation
- **🔍 Vector Search**: Semantic similarity search using Amazon Titan embeddings
- **📊 Database Queries**: Direct MongoDB queries for filtering, counting, and exact matches
- **🎯 Smart Tool Selection**: Automatically chooses between vector search and database queries
- **💬 Interactive UI**: Clean Streamlit web interface with real-time chat
- **🔧 Easy Configuration**: Simple setup through environment variables or UI

## 🏗️ Architecture

```
User Query → Claude (AWS Bedrock) → Tool Selection → MongoDB Operations → Results
                                  ↓
                          Vector Search (Titan) or Direct Query
```

## 📋 Prerequisites

- **AWS Account** with Bedrock access
- **MongoDB Atlas** cluster with vector search enabled
- **Python 3.9+**

### 1. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the project root:
```env
# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_SESSION_TOKEN=your_session_token  # Optional, for temporary credentials

# MongoDB
MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/
```

## 📁 Project Structure

```
/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .env           # # Environment variables (create this)
```

## ⚙️ AWS Bedrock Configuration Options

### Claude Models Available
- `anthropic.claude-3-sonnet-20240229-v1:0` (Recommended)
- `anthropic.claude-3-haiku-20240307-v1:0` (Faster/Cheaper)
- `anthropic.claude-instant-v1` (Legacy)
  
### Titan Models Available
- `amazon.titan-embed-text-v2:0` (Recommended)
- `amazon.titan-embed-text-v1` 

### AWS Regions Supported
- `us-east-1` (Most models available)
- `us-east-2`
- `us-west-2`
- `eu-west-1`
- `ap-southeast-1`
- `eu-central-1`

## 🗄️ MongoDB Atlas Setup

MongoDB collection with Titan Embeddings (default dimension used 1024)

## 🚀 Usage

### 1. Start the Application
```bash
streamlit run app.py
```
Debug Mode
Enable debug logging:
```bash
streamlit run app.py --logger.level=debug
```

### 2. Configure in Browser
1. Open `http://localhost:8501`
2. Enter AWS credentials in the sidebar
3. Configure MongoDB connection details
4. Click "Initialize Agent"
5. Wait until you get 🟢 Agent Ready 

### 3. Start Chatting
Ask questions like:
- **Vector Search**: "Find documents about ..."
- **Database Query**: "Count total documents in the collection"
- **Exploration**: "What information do you have about ...?"

## 💡 Query Examples

### Vector Search (Semantic Similarity)
```
✅ "Search for documents about ..."
✅ "Find documents related to ..."
✅ "What do you have about ...?"
✅ "Tell me about ..."
```

### Database Queries (Exact Matching)
```
✅ "Count total documents"
✅ "Find documents where category equals 'technology'"
✅ "How many documents were created in 2024?"
```

## 📝 License

This project is licensed under the MIT License
