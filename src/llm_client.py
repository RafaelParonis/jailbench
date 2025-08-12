import json
import os
from typing import Any, Dict, List, Optional, Tuple

import litellm


class LiteLLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        model_config: Optional[Dict] = None
    ):
        """
        Initialize the LiteLLM client.

        provider: str
            The LLM provider (e.g., 'openai', 'anthropic', 'google', 'azure')
        model: str
            The model display name (e.g., 'gpt-4', 'claude-3-sonnet')
        api_key: str
            The API key for the provider
        model_config: Optional[Dict]
            Model-specific configuration including litellm_model
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.model_config = model_config or {}

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using the configured LLM.

        messages: List[Dict[str, str]]
            List of message dictionaries with 'role' and 'content'
        max_tokens: Optional[int]
            Maximum tokens to generate
        temperature: Optional[float]
            Sampling temperature
        **kwargs
            Additional parameters to pass to the completion

        Returns
        -------
        str
            The generated response content
        """
        try:
            model_defaults = self.model_config
            completion_params = {
                "model": model_defaults.get("litellm_model", self.model),
                "messages": messages,
                "api_key": self.api_key,
                "max_tokens": max_tokens or model_defaults.get("max_tokens"),
                "temperature": temperature or model_defaults.get("temperature"),
                **kwargs
            }
            if self.provider == "azure":
                if "api_base" in model_defaults:
                    completion_params["api_base"] = model_defaults["api_base"]
                if "api_version" in model_defaults:
                    completion_params["api_version"] = model_defaults["api_version"]
            response = litellm.completion(**completion_params)
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def generate_response_with_usage(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, Any]:
        """
        Generate a response with full usage data and cost information.

        messages: List[Dict[str, str]]
            List of message dictionaries with 'role' and 'content'
        max_tokens: Optional[int]
            Maximum tokens to generate
        temperature: Optional[float]
            Sampling temperature
        **kwargs
            Additional parameters to pass to the completion

        Returns
        -------
        Tuple[str, Any]
            The generated response content and full LiteLLM response object
        """
        try:
            model_defaults = self.model_config
            completion_params = {
                "model": model_defaults.get("litellm_model", self.model),
                "messages": messages,
                "api_key": self.api_key,
                "max_tokens": max_tokens or model_defaults.get("max_tokens"),
                "temperature": temperature or model_defaults.get("temperature"),
                **kwargs
            }
            if self.provider == "azure":
                if "api_base" in model_defaults:
                    completion_params["api_base"] = model_defaults["api_base"]
                if "api_version" in model_defaults:
                    completion_params["api_version"] = model_defaults["api_version"]
            response = litellm.completion(**completion_params)
            return response.choices[0].message.content, response
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    @classmethod
    def from_credentials(
        cls,
        credentials_path: str = "credentials.json"
    ) -> List["LiteLLMClient"]:
        """
        Create multiple LiteLLMClient instances from credentials JSON file.

        credentials_path: str
            Path to credentials JSON file

        Returns
        -------
        List[LiteLLMClient]
            List of configured client instances for enabled models
        """
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        clients = []
        for provider_name, provider_data in credentials["providers"].items():
            api_key = provider_data["api_key"]
            provider_config = {k: v for k, v in provider_data.items() if k not in ["api_key", "models"]}
            for model_name, model_config in provider_data["models"].items():
                if model_config.get("enabled", False):
                    full_config = {**provider_config, **model_config}
                    client = cls(
                        provider=provider_name,
                        model=model_name,
                        api_key=api_key,
                        model_config=full_config
                    )
                    clients.append(client)
        return clients