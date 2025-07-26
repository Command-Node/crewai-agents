# main.py - Simple CrewAI server for Railway with OpenRouter
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Agent API")

# Simple in-memory storage for now (will upgrade to database later)
agents_storage = {}
personalities_storage = {}

class ChatRequest(BaseModel):
    message: str
    agent_id: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    agent_id: str
    session_id: str

class AgentPersonality(BaseModel):
    agent_id: str
    role: str
    goal: str
    backstory: str
    communication_style: str = "Friendly"
    temperature: float = 0.7

def create_agent(personality: AgentPersonality) -> Agent:
    """Create a CrewAI agent from personality"""
    llm = ChatOpenAI(
        model="meta-llama/llama-3.1-8b-instruct:free",  # Free model on OpenRouter
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=personality.temperature
    )
    
    enhanced_backstory = f"""
    {personality.backstory}
    
    Communication Style: {personality.communication_style}
    
    Always maintain this personality in your responses.
    """
    
    agent = Agent(
        role=personality.role,
        goal=personality.goal,
        backstory=enhanced_backstory,
        llm=llm,
        verbose=True
    )
    
    return agent

@app.get("/")
async def root():
    return {"message": "CrewAI Agent API is running!", "status": "healthy"}

@app.post("/create-agent")
async def create_agent_personality(personality: AgentPersonality):
    """Create a new agent with specific personality"""
    try:
        # Store the personality
        personalities_storage[personality.agent_id] = personality
        
        # Create the agent
        agent = create_agent(personality)
        agents_storage[personality.agent_id] = agent
        
        return {
            "message": f"Agent '{personality.agent_id}' created successfully!",
            "agent_id": personality.agent_id,
            "role": personality.role
        }
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """Chat with a specific agent"""
    try:
        # Get or create default agent
        if request.agent_id not in agents_storage:
            # Create a default agent if it doesn't exist
            default_personality = AgentPersonality(
                agent_id=request.agent_id,
                role="Helpful Assistant",
                goal="Help users with their questions effectively and friendly",
                backstory="You are a helpful AI assistant who is knowledgeable and friendly. You provide clear, useful answers.",
                communication_style="Friendly and professional"
            )
            personalities_storage[request.agent_id] = default_personality
            agents_storage[request.agent_id] = create_agent(default_personality)
        
        agent = agents_storage[request.agent_id]
        
        # Create a task for the conversation
        task = Task(
            description=f"Respond to this user message in a helpful way: {request.message}",
            agent=agent,
            expected_output="A helpful, clear response that matches your personality"
        )
        
        # Create crew and get response
        crew = Crew(
            agents=[agent],
            tasks=[task]
        )
        
        result = crew.kickoff()
        
        return ChatResponse(
            response=str(result),
            agent_id=request.agent_id,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            agent_id=request.agent_id,
            session_id=request.session_id
        )

@app.get("/agents")
async def list_agents():
    """List all created agents"""
    agents_list = []
    for agent_id, personality in personalities_storage.items():
        agents_list.append({
            "agent_id": agent_id,
            "role": personality.role,
            "goal": personality.goal,
            "communication_style": personality.communication_style
        })
    return {"agents": agents_list}

@app.get("/agents/{agent_id}")
async def get_agent_info(agent_id: str):
    """Get information about a specific agent"""
    if agent_id not in personalities_storage:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    personality = personalities_storage[agent_id]
    return {
        "agent_id": agent_id,
        "role": personality.role,
        "goal": personality.goal,
        "backstory": personality.backstory,
        "communication_style": personality.communication_style,
        "temperature": personality.temperature
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
