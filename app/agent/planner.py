"""
AI Agent Planner — ReAct döngüsü ile çalışan ana agent.
LLM + Tools + Memory entegrasyonu.
"""
from __future__ import annotations

import logging  # Uygulama loglama
from dataclasses import dataclass, field  # Veri sınıfı tanımı için

from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama  # Yerel LLM entegrasyonu
from langchain_core.prompts import PromptTemplate  # Prompt şablonu oluşturmak için
from langchain.memory import ConversationBufferWindowMemory

from app.agent.tools import ALL_TOOLS
from app.core.config import settings

logger = logging.getLogger(__name__)

# ReAct Prompt Template
# Agent'a nasıl düşüneceğini, hangi formatı kullanacağını söyler
REACT_PROMPT = PromptTemplate.from_template(
    """Sen AI Commerce Assistant'sın. Kullanıcılara e-ticaret sorularında yardım edersin.
Elindeki araçları kullanarak adım adım düşün ve en doğru yanıtı ver.
Türkçe konuş.

Elindeki araçlar:
{tools}

Araç adları: {tool_names}

Geçmiş konuşma:
{chat_history}

Kullanıcı sorusu: {input}

Aşağıdaki formatı KULLAN:

Thought: [Ne yapmalıyım? Hangi aracı kullanmalıyım?]
Action: [araç_adı]
Action Input: [araç_girdisi]
Observation: [araç çıktısı]
... (Bu döngü tekrarlanabilir)
Thought: Artık yanıt verebilecek kadar bilgim var.
Final Answer: [Kullanıcıya verilecek Türkçe yanıt]

Action Input her zaman SADECE düz metin olmalıdır.
JSON, dict veya parametre formatı kullanma.

tool_names de olmayan bir araç kullanma, eğer kullanman gereken araç yoksa "Bu araç mevcut değil" yaz.

Başla!
{agent_scratchpad}"""
)


@dataclass
class AgentResponse:
    """Agent yanıt modeli."""
    answer: str
    steps: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    
    
class CommerceAgent:
    """
    E-ticaret asistanı agent'ı.
    ReAct pattern ile çalışır, araçları kullanarak sorulara yanıt verir.
    """

    def __init__(self) -> None:
        self._executor: AgentExecutor | None = None
        self._memory = ConversationBufferWindowMemory(
            k=5,                         # Son 5 konuşmayı hatırla
            memory_key="chat_history",
            return_messages=False,
        )
        
    def _build_executor(self) -> AgentExecutor:
        """Agent executor'ı oluşturur (lazy loading)."""
        llm = Ollama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,
            num_predict=1024,
        )

        agent = create_react_agent(
            llm=llm,
            tools=ALL_TOOLS,
            prompt=REACT_PROMPT,
        )

        return AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            memory=self._memory,
            verbose=True,                # Adımları logla
            max_iterations=5,            # Maksimum döngü sayısı (sonsuz döngüyü önler)
            handle_parsing_errors=True,  # LLM format hatalarını tolere et
            return_intermediate_steps=True,
        )    
        
    async def run(self, user_input: str) -> AgentResponse:
        """
        Kullanıcı girdisini agent ile işler.

        Args:
            user_input: Kullanıcının sorusu

        Returns:
            AgentResponse: Yanıt ve kullanılan araçlar
        """
        if self._executor is None:
            self._executor = self._build_executor()

        logger.info("Agent çalışıyor: %s", user_input)
        
        try:
            result = await self._executor.ainvoke({"input": user_input})
        except Exception as e:
            logger.exception("Agent hatası: %s", e)
            return AgentResponse(
                answer="Üzgünüm, isteğinizi işlerken bir hata oluştu. Lütfen tekrar deneyin.",
            )
            
        # Kullanılan araçları çıkar
        tools_used = []
        steps = []
        for action, observation in result.get("intermediate_steps", []):
            tools_used.append(action.tool)
            steps.append({
                "tool": action.tool,
                "input": action.tool_input,
                "output": str(observation)[:200],
            })

        return AgentResponse(
            answer=result.get("output", ""),
            steps=steps,
            tools_used=list(set(tools_used)),
        )  
        
    def clear_memory(self) -> None:
        """Konuşma geçmişini temizler."""
        self._memory.clear()

# Singleton
commerce_agent = CommerceAgent()        
        
            