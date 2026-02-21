# agents/brief_generator.py
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
from pathlib import Path

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv()

class MasterBriefGenerator:
    """RAG-based Master's Brief Generator"""
    
    def __init__(self, data_dir: str = "data", model: str = "llama3.1"):
        self.data_dir = Path(data_dir)
        self.model = model
        
        # Initialize Ollama (FREE, local)
        print(f"🤖 Using Ollama with model: {model}")
        
        try:
            self.llm = Ollama(
                model=model,
                temperature=0.3
            )
            
            # Test if model is available
            test_response = self.llm.invoke("test")
            print(f"✅ Model {model} loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model {model}: {e}")
            print(f"\n💡 Available models on your system:")
            os.system("ollama list")
            print(f"\n📥 To download this model, run:")
            print(f"   ollama pull {model}")
            raise
        
        # Use Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=model
        )
        
        # Initialize or load vector store
        self.vectorstore = self._setup_vectorstore()
        
        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
    
    def _setup_vectorstore(self):
        """Initialize ChromaDB with company documents"""
        persist_dir = str(self.data_dir / "vector_db")
        
        # Check if vector store already exists
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print("📚 Loading existing vector store...")
            try:
                return Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"⚠️  Error loading vector store: {e}")
                print("🔄 Recreating vector store...")
                import shutil
                shutil.rmtree(persist_dir)
        
        print("📚 Creating new vector store from company documents...")
        
        # Load company documents
        docs = []
        docs_dir = self.data_dir / "company_docs"
        
        if not docs_dir.exists():
            print(f"⚠️  Directory {docs_dir} not found. Creating it...")
            docs_dir.mkdir(parents=True, exist_ok=True)
        
        for doc_file in docs_dir.glob("*.txt"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": doc_file.name,
                        "type": "company_procedure"
                    }
                ))
        
        if not docs:
            print("⚠️  No company documents found. Creating placeholder...")
            docs = [Document(
                page_content="Maritime operations placeholder document.",
                metadata={"source": "placeholder", "type": "placeholder"}
            )]
        
        # Create vector store
        print("🔄 Creating embeddings... (this may take 30-60 seconds)")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        
        print(f"✅ Indexed {len(docs)} company documents")
        return vectorstore
    
    def load_sample_data(self) -> Dict:
        """Load all sample data files"""
        sample_dir = self.data_dir / "sample_data"
        
        # Load CSV
        maintenance_df = pd.read_csv(sample_dir / "maintenance_tasks.csv")
        
        # Load JSONs
        with open(sample_dir / "weather_forecast.json") as f:
            weather_data = json.load(f)
        
        with open(sample_dir / "security_alerts.json") as f:
            security_data = json.load(f)
        
        with open(sample_dir / "voyage_plan.json") as f:
            voyage_data = json.load(f)
        
        return {
            "maintenance": maintenance_df,
            "weather": weather_data,
            "security": security_data,
            "voyage": voyage_data
        }
    
    def generate_brief(self, vessel_id: str = "MV-DEMO-001") -> str:
        """Generate complete Master's Brief"""
        
        print(f"\n🚢 Generating brief for {vessel_id}...")
        
        # Load data
        data = self.load_sample_data()
        
        # Prepare context
        context = self._prepare_context(data)
        
        # Create prompt
        prompt = self._create_brief_prompt(context, vessel_id)
        
        # Generate brief using LLM
        print(f"🤖 Generating brief with {self.model}...")
        print("⏳ This may take 60-120 seconds on CPU...")
        
        response = self.llm.invoke(prompt)
        brief_content = response
        
        # Enhance with RAG retrieval for specific sections
        brief_content = self._enhance_with_rag(brief_content, data)
        
        print("✅ Brief generated successfully!\n")
        
        return brief_content
    
    def _prepare_context(self, data: Dict) -> Dict:
        """Prepare structured context from data"""
        
        # Process maintenance tasks
        maintenance_df = data["maintenance"]
        critical_tasks = maintenance_df[
            maintenance_df["priority"].isin(["CRITICAL", "HIGH"])
        ].to_dict('records')
        
        # Process weather alerts
        weather_alerts = []
        for wp in data["weather"]["waypoints"]:
            if wp.get("alerts"):
                weather_alerts.extend(wp["alerts"])
        
        # Process security alerts
        high_risk_alerts = [
            alert for alert in data["security"]["alerts"]
            if alert["risk_level"] in ["HIGH", "CRITICAL"]
        ]
        
        return {
            "vessel_name": data["voyage"]["vessel_name"],
            "voyage_number": data["voyage"]["voyage_number"],
            "route": data["voyage"]["route_summary"][0],
            "departure_date": data["voyage"]["departure_date"],
            "eta": data["voyage"]["eta_destination"],
            "critical_tasks": critical_tasks,
            "all_tasks": maintenance_df.to_dict('records'),
            "weather_alerts": weather_alerts,
            "weather_waypoints": data["weather"]["waypoints"],
            "security_alerts": high_risk_alerts,
            "all_security": data["security"]["alerts"],
            "critical_dates": data["voyage"]["critical_dates"]
        }
    
    def _create_brief_prompt(self, context: Dict, vessel_id: str) -> str:
        """Create the main prompt for brief generation"""
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""You are a Maritime Operations AI Assistant preparing the Weekly Master's Brief.
You MUST follow maritime industry standards and use professional terminology.

IMPORTANT FORMATTING RULES:
- Use clear section headers with ===
- Include specific action deadlines
- Highlight CRITICAL items with 🔴
- Use tables for maintenance schedules
- Reference company procedure numbers (e.g., SMS-SEC-001)

VESSEL INFORMATION:
- Vessel: {context['vessel_name']} ({vessel_id})
- Voyage: {context['voyage_number']}
- Route: {context['route']}
- Departure: {context['departure_date']}
- ETA Destination: {context['eta']}
- Current Date: {current_date}

CRITICAL MAINTENANCE TASKS (Next 14 Days):
{self._format_tasks(context['critical_tasks'][:5])}

WEATHER HIGHLIGHTS:
{self._format_weather(context['weather_waypoints'])}

SECURITY ALERTS:
{self._format_security(context['all_security'])}

CRITICAL VOYAGE DATES:
{self._format_critical_dates(context['critical_dates'])}

Generate a professional Master's Brief with these sections:

1. EXECUTIVE SUMMARY (2-3 sentences)
2. CRITICAL ALERTS
3. UPCOMING MAINTENANCE (Next 14 Days)
4. ROUTE WEATHER ANALYSIS
5. SECURITY BRIEFING
6. RECOMMENDED ACTIONS

Use professional maritime terminology. Be concise but comprehensive.
"""
        return prompt
    
    def _enhance_with_rag(self, brief: str, data: Dict) -> str:
        """Enhance brief with RAG retrieval for specific procedures"""
        
        try:
            # Check if Gulf of Aden mentioned
            if "Gulf of Aden" in brief or "piracy" in brief.lower():
                print("📖 Retrieving anti-piracy procedures...")
                piracy_info = self.qa_chain.invoke({
                    "query": "What are the mandatory anti-piracy measures for Gulf of Aden transit?"
                })
                
                brief += "\n\n---\n\n### APPENDIX A: Anti-Piracy Procedure Reference\n\n"
                brief += str(piracy_info.get("result", ""))
        except Exception as e:
            print(f"⚠️  Could not retrieve additional context: {e}")
        
        return brief
    
    # Helper formatting methods
    def _format_tasks(self, tasks: List[Dict]) -> str:
        if not tasks:
            return "No critical tasks in the next 14 days."
        
        lines = []
        for task in tasks:
            lines.append(
                f"- [{task['priority']}] {task['equipment']}: {task['description']} "
                f"(Due: {task['due_date']}, {task['estimated_hours']}hrs)"
            )
        return "\n".join(lines)
    
    def _format_weather(self, waypoints: List[Dict]) -> str:
        lines = []
        for wp in waypoints[:3]:
            forecast = wp["forecast"]
            alerts = wp.get("alerts", [])
            alert_str = f" ⚠️ {alerts[0]}" if alerts else ""
            
            lines.append(
                f"- {wp['location']} (ETA {wp['eta']}): "
                f"{forecast['weather']}, Wind {forecast['wind_speed_kts']}kts, "
                f"Waves {forecast['wave_height_m']}m{alert_str}"
            )
        return "\n".join(lines)
    
    def _format_security(self, alerts: List[Dict]) -> str:
        if not alerts:
            return "No current security alerts for route."
        
        lines = []
        for alert in alerts:
            lines.append(
                f"- [{alert['risk_level']}] {alert['region']}: {alert['title']}"
            )
        return "\n".join(lines)
    
    def _format_critical_dates(self, dates: List[Dict]) -> str:
        lines = []
        for item in dates:
            lines.append(f"- {item['date']}: {item['event']}")
        return "\n".join(lines)


# Test function
if __name__ == "__main__":
    generator = MasterBriefGenerator(model="llama3.1")  # Change model here
    brief = generator.generate_brief()
    print("\n" + "="*80)
    print(brief)
    print("="*80)