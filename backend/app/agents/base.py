"""Base agent class for incident analysis agents."""
import json
import structlog
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Abstract base class for all incident analysis agents.
    
    Each agent:
    1. Takes structured input data
    2. Generates a prompt using a template
    3. Calls an LLM
    4. Validates output against a Pydantic schema
    5. Returns structured output
    """
    
    name: str = "base"
    output_schema: Type[BaseModel] = BaseModel
    
    def __init__(self):
        self.model = settings.OPENAI_MODEL
        self.api_key = settings.OPENAI_API_KEY
        self.max_retries = settings.AGENT_MAX_RETRIES
    
    @abstractmethod
    def get_prompt(self, input_data: Dict[str, Any]) -> str:
        """Generate the prompt for this agent."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the given input data.
        
        Args:
            input_data: Input data for the agent
        
        Returns:
            Validated output dictionary
        """
        logger.info(f"Running agent", agent=self.name)
        
        # Check if we have an API key
        if not self.api_key:
            logger.warning("No OpenAI API key, using mock response", agent=self.name)
            return self._get_mock_response(input_data)
        
        try:
            # Generate prompt
            prompt = self.get_prompt(input_data)
            system_prompt = self.get_system_prompt()
            
            # Call LLM with retry
            response = self._call_llm_with_retry(system_prompt, prompt)
            
            # Parse and validate response
            result = self._parse_response(response)
            
            return result
        
        except Exception as e:
            logger.error(f"Agent execution failed", agent=self.name, error=str(e))
            return self._get_mock_response(input_data)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM with retry logic."""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Lower temperature for more deterministic output
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate the LLM response."""
        try:
            data = json.loads(response)
            
            # Validate against schema
            validated = self.output_schema.model_validate(data)
            return validated.model_dump()
        
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON response", agent=self.name, error=str(e))
            # Try to fix the response
            return self._try_fix_response(response)
        
        except ValidationError as e:
            logger.error("Schema validation failed", agent=self.name, error=str(e))
            # Return partial data
            try:
                return json.loads(response)
            except:
                return {"error": "Validation failed", "raw": response[:500]}
    
    def _try_fix_response(self, response: str) -> Dict[str, Any]:
        """Attempt to fix invalid JSON response."""
        # Simple fixes
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        try:
            return json.loads(response.strip())
        except:
            return {"error": "Could not parse response", "raw": response[:500]}
    
    @abstractmethod
    def _get_mock_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get a mock response when LLM is not available."""
        pass

