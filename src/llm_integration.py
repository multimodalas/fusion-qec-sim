"""
LLM Integration Module for IRC Bot

Provides conversational AI capabilities using LLM APIs:
- Code generation
- Simulation explanation
- Chat moderation
- Creative coding assistance

Supports various LLM providers with rate limiting and ethical use controls.
"""

import time
import json
from typing import Optional, Dict, List, Callable
from datetime import datetime, timedelta
import os


class RateLimiter:
    """
    Rate limiter for API calls.
    """
    
    def __init__(self, calls_per_minute: int = 10):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum API calls per minute
        """
        self.calls_per_minute = calls_per_minute
        self.call_times: List[datetime] = []
        
    def can_proceed(self) -> bool:
        """
        Check if new call can proceed.
        
        Returns:
            True if within rate limit
        """
        now = datetime.now()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times 
                          if now - t < timedelta(minutes=1)]
        
        # Check if under limit
        return len(self.call_times) < self.calls_per_minute
    
    def record_call(self):
        """Record a new API call."""
        self.call_times.append(datetime.now())
    
    def wait_if_needed(self):
        """Wait if rate limit reached."""
        while not self.can_proceed():
            time.sleep(1)


class LLMProvider:
    """
    Base class for LLM providers.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = 10,
        max_tokens: int = 500
    ):
        """
        Initialize LLM provider.
        
        Args:
            api_key: API key for LLM service
            rate_limit: Calls per minute
            max_tokens: Maximum response tokens
        """
        self.api_key = api_key or os.environ.get('LLM_API_KEY', '')
        self.rate_limiter = RateLimiter(rate_limit)
        self.max_tokens = max_tokens
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        raise NotImplementedError("Subclass must implement generate()")


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for testing and demo purposes.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize mock provider."""
        super().__init__(*args, **kwargs)
        
        # Predefined responses for common queries
        self.responses = {
            'explain': (
                "The Steane [[7,1,3]] code is a quantum error correction code "
                "that encodes 1 logical qubit into 7 physical qubits. It can "
                "correct any single-qubit error and has a distance of 3. "
                "The pseudo-threshold is approximately η_thr ≈ 9.3×10⁻⁵."
            ),
            'code': (
                "Here's a simple example:\n"
                "```python\n"
                "from qec_steane import SteaneCode\n"
                "code = SteaneCode()\n"
                "state = code.encode_logical_zero()\n"
                "noisy = code.apply_depolarizing_noise(state, p=0.01)\n"
                "```"
            ),
            'threshold': (
                "The pseudo-threshold is the physical error rate below which "
                "error correction improves the logical error rate. For Steane "
                "code, it's around 9.3×10⁻⁵. Below this rate, encoding helps; "
                "above it, encoding can make things worse."
            ),
            'midi': (
                "MIDI export maps simulation data to music: physical error rates "
                "become tempo (0.01 → 120 BPM), logical errors trigger e-minor "
                "arpeggios, and eigenvalues map to note velocities [8, 92]. "
                "This creates an auditory representation of quantum error dynamics."
            )
        }
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate mock response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Mock response
        """
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        self.rate_limiter.record_call()
        
        # Simple keyword matching
        prompt_lower = prompt.lower()
        
        if 'explain' in prompt_lower or 'what is' in prompt_lower:
            response = self.responses['explain']
        elif 'code' in prompt_lower or 'example' in prompt_lower:
            response = self.responses['code']
        elif 'threshold' in prompt_lower:
            response = self.responses['threshold']
        elif 'midi' in prompt_lower or 'music' in prompt_lower:
            response = self.responses['midi']
        else:
            response = (
                "I'm a QEC bot powered by LLM. I can explain quantum error "
                "correction concepts, generate code examples, and help with "
                "simulations. Ask me about Steane codes, thresholds, or MIDI export!"
            )
        
        return response


class LLMChatBot:
    """
    LLM-powered chatbot for QEC discussions.
    """
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLM chatbot.
        
        Args:
            provider: LLM provider instance
            system_prompt: System prompt for bot personality
        """
        self.provider = provider or MockLLMProvider()
        
        self.system_prompt = system_prompt or (
            "You are a helpful quantum error correction expert bot. "
            "You can explain QEC concepts, generate code examples using QuTiP, "
            "discuss threshold calculations, and help users understand MIDI exports "
            "of quantum simulation data. Be concise, accurate, and friendly. "
            "For code generation, use the qec_steane and midi_export modules."
        )
        
        # Conversation history
        self.history: List[Dict[str, str]] = []
        
        # Content moderation
        self.blocked_words = ['spam', 'abuse']  # Minimal example
        
    def should_respond(self, message: str) -> bool:
        """
        Determine if bot should respond to message.
        
        Args:
            message: User message
            
        Returns:
            True if bot should respond
        """
        # Check for blocked content
        message_lower = message.lower()
        if any(word in message_lower for word in self.blocked_words):
            return False
        
        # Check if message is directed at bot
        # (In IRC context, this would check mentions or direct messages)
        return True
    
    def generate_response(
        self,
        message: str,
        user: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate response to user message.
        
        Args:
            message: User message
            user: Username (optional)
            
        Returns:
            Bot response or None
        """
        if not self.should_respond(message):
            return None
        
        # Add to history
        self.history.append({
            'user': user or 'unknown',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response
        try:
            response = self.provider.generate(
                prompt=message,
                system_prompt=self.system_prompt
            )
            
            # Add response to history
            self.history.append({
                'user': 'bot',
                'message': response,
                'timestamp': datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error processing your request."
    
    def generate_code(self, description: str) -> str:
        """
        Generate code based on description.
        
        Args:
            description: Description of desired code
            
        Returns:
            Generated code
        """
        prompt = (
            f"Generate Python code using QuTiP for: {description}\n"
            "Use the qec_steane module if available."
        )
        
        return self.provider.generate(prompt, self.system_prompt)
    
    def explain_simulation(self, sim_data: Dict) -> str:
        """
        Explain simulation results.
        
        Args:
            sim_data: Simulation data dictionary
            
        Returns:
            Explanation text
        """
        prompt = (
            f"Explain these QEC simulation results:\n"
            f"{json.dumps(sim_data, indent=2)}"
        )
        
        return self.provider.generate(prompt, self.system_prompt)
    
    def moderate_message(self, message: str) -> Dict[str, bool]:
        """
        Moderate message content.
        
        Args:
            message: Message to moderate
            
        Returns:
            Moderation result dict
        """
        result = {
            'allowed': True,
            'reason': None
        }
        
        message_lower = message.lower()
        
        # Check blocked words
        for word in self.blocked_words:
            if word in message_lower:
                result['allowed'] = False
                result['reason'] = 'Contains blocked content'
                break
        
        # Check message length
        if len(message) > 500:
            result['allowed'] = False
            result['reason'] = 'Message too long'
        
        return result


def demo_llm_integration():
    """
    Demonstrate LLM integration functionality.
    """
    print("=== LLM Integration Demo ===\n")
    
    # Initialize mock provider
    print("Initializing mock LLM provider...")
    provider = MockLLMProvider(rate_limit=10)
    
    # Initialize chatbot
    print("Initializing LLM chatbot...")
    bot = LLMChatBot(provider)
    
    print("\n--- Demo 1: Simple Conversation ---")
    test_queries = [
        "What is the Steane code?",
        "Show me example code",
        "Explain the threshold",
        "How does MIDI export work?"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = bot.generate_response(query, user="demo_user")
        print(f"Bot: {response}")
    
    print("\n--- Demo 2: Code Generation ---")
    description = "Simulate Steane code with depolarizing noise"
    print(f"\nRequest: {description}")
    code = bot.generate_code(description)
    print(f"Generated:\n{code}")
    
    print("\n--- Demo 3: Content Moderation ---")
    test_messages = [
        "Help me understand QEC",
        "This is spam message",
        "Tell me about thresholds"
    ]
    
    for msg in test_messages:
        moderation = bot.moderate_message(msg)
        status = "✓ Allowed" if moderation['allowed'] else "✗ Blocked"
        print(f"\n{status}: {msg}")
        if not moderation['allowed']:
            print(f"  Reason: {moderation['reason']}")
    
    print("\n--- Demo 4: Rate Limiting ---")
    limiter = RateLimiter(calls_per_minute=3)
    
    print("\nMaking 5 rapid calls (limit: 3/min):")
    for i in range(5):
        if limiter.can_proceed():
            limiter.record_call()
            print(f"  Call {i+1}: ✓ Allowed")
        else:
            print(f"  Call {i+1}: ✗ Rate limit reached")
    
    print("\n=== Demo Complete ===")


if __name__ == '__main__':
    demo_llm_integration()
