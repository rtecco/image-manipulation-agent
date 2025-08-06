"""Token-aware rate limiter for Anthropic API."""

import time
import threading
from typing import Optional, List

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage


class TokenAwareRateLimiter:
    """Token bucket rate limiter for Anthropic API based on input tokens per minute."""
    
    def __init__(
        self,
        tokens_per_minute: int = 40000,  # Conservative default
        max_burst_tokens: Optional[int] = None,
        anthropic_client: Optional[anthropic.Anthropic] = None
    ):
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_second = tokens_per_minute / 60.0
        self.max_burst_tokens = max_burst_tokens or tokens_per_minute // 4
        
        self.available_tokens = self.max_burst_tokens
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        # For token counting - will try to create client with API key from env
        try:
            self.anthropic_client = anthropic_client or anthropic.Anthropic()
        except Exception:
            self.anthropic_client = None
    
    def _refill_tokens(self) -> None:
        """Refill the token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.tokens_per_second
        
        self.available_tokens = min(
            self.max_burst_tokens, 
            self.available_tokens + new_tokens
        )
        self.last_refill = now
    
    def count_message_tokens(self, messages: List[BaseMessage], model: str) -> int:
        """Count tokens for a list of LangChain messages."""
        if not self.anthropic_client:
            # Fallback: rough estimation (4 chars per token)
            total_chars = sum(len(str(msg.content)) for msg in messages if hasattr(msg, 'content'))
            return max(10, total_chars // 4)
            
        try:
            # Convert LangChain messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                        anthropic_messages.append({
                            "role": role,
                            "content": msg.content
                        })
                    elif isinstance(msg.content, list):
                        # Handle multimodal content (images + text)
                        role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                        content_parts = []
                        for part in msg.content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    content_parts.append({"type": "text", "text": part.get("text", "")})
                                elif part.get("type") == "image":
                                    # Add significant token count for images
                                    content_parts.append({"type": "text", "text": "[IMAGE_PLACEHOLDER]"})
                        if content_parts:
                            anthropic_messages.append({
                                "role": role, 
                                "content": content_parts
                            })
            
            if not anthropic_messages:
                return 10
                
            token_count = self.anthropic_client.messages.count_tokens(
                model=model,
                messages=anthropic_messages
            )
            return token_count.input_tokens
            
        except Exception:
            # Fallback: rough estimation with image penalty
            total_chars = 0
            image_count = 0
            
            for msg in messages:
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, str):
                        total_chars += len(msg.content)
                    elif isinstance(msg.content, list):
                        for part in msg.content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    total_chars += len(part.get("text", ""))
                                elif part.get("type") == "image":
                                    image_count += 1
            
            base_tokens = max(10, total_chars // 4)
            image_tokens = image_count * 1500  # ~1500 tokens per image
            return base_tokens + image_tokens
    
    def acquire_tokens(self, token_count: int, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from the bucket. Waits until tokens are available."""
        start_time = time.time()
        
        # Check if request exceeds max burst capacity
        if token_count > self.max_burst_tokens:
            print(f"🔄 Request needs {token_count} tokens but max burst is {self.max_burst_tokens}. Increasing burst capacity.")
            self.max_burst_tokens = token_count + 1000  # Add some buffer
            self.available_tokens = min(self.available_tokens, self.max_burst_tokens)
        
        while True:
            with self.lock:
                self._refill_tokens()
                
                if self.available_tokens >= token_count:
                    self.available_tokens -= token_count
                    print(f"✅ Acquired {token_count} tokens. {self.available_tokens:.0f} remaining.")
                    return True
                
                # Calculate how long to wait for the required tokens
                tokens_needed = token_count - self.available_tokens
                wait_time = tokens_needed / self.tokens_per_second
            
            # Check timeout only if specified
            if timeout is not None and time.time() - start_time > timeout:
                return False
            
            # Sleep for the calculated wait time, but wake up periodically to check
            sleep_time = min(wait_time, 2.0)  # Sleep up to 2 seconds at a time
            print(f"⏳ Need {tokens_needed} more tokens. Waiting {sleep_time:.1f}s... (available: {self.available_tokens:.0f}/{self.max_burst_tokens})")
            time.sleep(sleep_time)


class TokenAwareChatAnthropic:
    """Wrapper for ChatAnthropic with token-aware rate limiting."""
    
    def __init__(self, token_rate_limiter: TokenAwareRateLimiter, **kwargs):
        self.token_rate_limiter = token_rate_limiter
        self.llm = ChatAnthropic(**kwargs)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped ChatAnthropic instance."""
        return getattr(self.llm, name)
        
    def invoke(self, input_messages, **kwargs):
        """Override invoke to apply token-based rate limiting."""
        if self.token_rate_limiter:
            # Convert input to messages if needed
            messages = input_messages if isinstance(input_messages, list) else [input_messages]
            
            # Count tokens for the request
            token_count = self.token_rate_limiter.count_message_tokens(
                messages, 
                self.llm.model
            )
            
            print(f"🔢 Request requires ~{token_count} input tokens")
            
            # Wait for tokens to be available
            if not self.token_rate_limiter.acquire_tokens(token_count):
                raise Exception(f"Token rate limit exceeded. Request needs {token_count} tokens.")
        
        return self.llm.invoke(input_messages, **kwargs)