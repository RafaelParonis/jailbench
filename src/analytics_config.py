from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

@dataclass
class SemanticAnalysisConfig:
    """
    Configuration for semantic analysis of prompts and responses.

    enabled: bool
        Whether to run semantic analysis
    min_word_frequency: int
        Minimum word frequency for analysis
    cluster_threshold: float
        Similarity threshold for clustering
    analyze_successful_only: bool
        Only analyze successful jailbreaks
    extract_linguistic_features: bool
        Extract advanced linguistic features
    """
    enabled: bool = True
    min_word_frequency: int = 2
    cluster_threshold: float = 0.8
    analyze_successful_only: bool = False
    extract_linguistic_features: bool = True

@dataclass
class DefenseProfilingConfig:
    """
    Configuration for defense strength profiling and analysis.

    enabled: bool
        Whether to analyze defense patterns
    confidence_scoring: bool
        Score refusal confidence strength
    consistency_analysis: bool
        Analyze consistency across similar prompts
    response_length_analysis: bool
        Analyze response length patterns
    trigger_word_detection: bool
        Detect safety trigger words
    """
    enabled: bool = True
    confidence_scoring: bool = True
    consistency_analysis: bool = True
    response_length_analysis: bool = True
    trigger_word_detection: bool = True

@dataclass
class VulnerabilityMatrixConfig:
    """
    Configuration for cross-model vulnerability analysis.

    enabled: bool
        Whether to build vulnerability matrix
    create_heatmap: bool
        Generate attack technique vs model heatmap
    model_fingerprinting: bool
        Create model vulnerability fingerprints
    shared_pattern_analysis: bool
        Find shared vulnerability patterns
    """
    enabled: bool = True
    create_heatmap: bool = True
    model_fingerprinting: bool = True
    shared_pattern_analysis: bool = True

@dataclass
class AttackVectorConfig:
    """
    Configuration for attack vector effectiveness analysis.

    enabled: bool
        Whether to analyze attack vectors
    length_correlation: bool
        Analyze prompt length vs success
    emotional_analysis: bool
        Analyze emotional appeal effectiveness
    structure_analysis: bool
        Analyze prompt structure patterns  
    escalation_tracking: bool
        Track multi-step escalation patterns
    """
    enabled: bool = True
    length_correlation: bool = True
    emotional_analysis: bool = True
    structure_analysis: bool = True
    escalation_tracking: bool = True

@dataclass
class ConversationFlowConfig:
    """
    Configuration for conversation flow analysis.

    enabled: bool
        Whether to analyze conversation flows
    persuasion_tracking: bool
        Track persuasion escalation patterns
    context_exploitation: bool
        Analyze context window exploitation
    memory_consistency: bool
        Analyze memory/consistency attacks
    trajectory_clustering: bool
        Cluster conversation trajectories
    """
    enabled: bool = True
    persuasion_tracking: bool = True
    context_exploitation: bool = True
    memory_consistency: bool = True
    trajectory_clustering: bool = True

@dataclass
class ResponsePatternConfig:
    """
    Configuration for response pattern intelligence.

    enabled: bool
        Whether to analyze response patterns
    pre_compliance_detection: bool
        Detect pre-compliance indicators
    information_leakage: bool
        Detect partial information leakage
    hedging_analysis: bool
        Analyze response hedging patterns
    refusal_classification: bool
        Classify refusal vs explanation patterns
    """
    enabled: bool = True
    pre_compliance_detection: bool = True
    information_leakage: bool = True
    hedging_analysis: bool = True
    refusal_classification: bool = True

@dataclass
class ToolCallingConfig:
    """
    Configuration for tool calling jailbreak tests.

    enabled: bool
        Whether to test tool calling vulnerabilities
    malicious_tool_calls: bool
        Test malicious tool call attempts
    tool_chaining: bool
        Test tool chaining for complex attacks
    permission_escalation: bool
        Test permission escalation via tools
    data_extraction: bool
        Test data extraction via tool calls
    """
    enabled: bool = False
    malicious_tool_calls: bool = False
    tool_chaining: bool = False
    permission_escalation: bool = False
    data_extraction: bool = False

@dataclass
class PromptOptimizationConfig:
    """
    Configuration for prompt engineering insights.

    enabled: bool
        Whether to analyze prompt optimization
    token_position_analysis: bool
        Analyze token position importance
    structure_templates: bool
        Extract successful prompt templates
    character_limit_analysis: bool
        Analyze character limits vs effectiveness
    component_analysis: bool
        Analyze successful prompt components
    """
    enabled: bool = True
    token_position_analysis: bool = True
    structure_templates: bool = True
    character_limit_analysis: bool = True
    component_analysis: bool = True

@dataclass
class PerformanceConfig:
    """
    Configuration for performance monitoring and optimization.

    enabled: bool
        Whether to monitor performance
    max_execution_time: float
        Max execution time per module in seconds
    memory_limit_mb: int
        Memory limit per module in MB
    parallel_processing: bool
        Enable parallel processing where possible
    cache_results: bool
        Cache intermediate results
    """
    enabled: bool = True
    max_execution_time: float = 30.0
    memory_limit_mb: int = 512
    parallel_processing: bool = True
    cache_results: bool = True

@dataclass
class AdvancedAnalyticsConfig:
    """
    Master configuration for all advanced analytics modules.

    semantic_analysis: SemanticAnalysisConfig
        Semantic analysis settings
    defense_profiling: DefenseProfilingConfig
        Defense profiling settings
    vulnerability_matrix: VulnerabilityMatrixConfig
        Vulnerability analysis settings
    attack_vectors: AttackVectorConfig
        Attack vector analysis settings
    conversation_flow: ConversationFlowConfig
        Conversation flow analysis settings
    response_patterns: ResponsePatternConfig
        Response pattern analysis settings
    tool_calling: ToolCallingConfig
        Tool calling test settings
    prompt_optimization: PromptOptimizationConfig
        Prompt optimization settings
    performance: PerformanceConfig
        Performance monitoring settings
    """
    semantic_analysis: SemanticAnalysisConfig = field(default_factory=SemanticAnalysisConfig)
    defense_profiling: DefenseProfilingConfig = field(default_factory=DefenseProfilingConfig)
    vulnerability_matrix: VulnerabilityMatrixConfig = field(default_factory=VulnerabilityMatrixConfig)
    attack_vectors: AttackVectorConfig = field(default_factory=AttackVectorConfig)
    conversation_flow: ConversationFlowConfig = field(default_factory=ConversationFlowConfig)
    response_patterns: ResponsePatternConfig = field(default_factory=ResponsePatternConfig)
    tool_calling: ToolCallingConfig = field(default_factory=ToolCallingConfig)
    prompt_optimization: PromptOptimizationConfig = field(default_factory=PromptOptimizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_file(cls, config_path: Path) -> 'AdvancedAnalyticsConfig':
        """
        Load configuration from JSON file.

        config_path: Path
            Path to configuration file

        Returns
        -------
        AdvancedAnalyticsConfig
            Loaded configuration
        """
        if not config_path.exists():
            return cls()
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

    def to_file(self, config_path: Path) -> None:
        """
        Save configuration to JSON file.

        config_path: Path
            Path to save configuration file
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=lambda x: x.__dict__)

    def get_enabled_modules(self) -> List[str]:
        """
        Get list of enabled analytics modules.

        Returns
        -------
        List[str]
            Names of enabled modules
        """
        enabled_modules = []
        if self.semantic_analysis.enabled:
            enabled_modules.append('semantic_analysis')
        if self.defense_profiling.enabled:
            enabled_modules.append('defense_profiling')
        if self.vulnerability_matrix.enabled:
            enabled_modules.append('vulnerability_matrix')
        if self.attack_vectors.enabled:
            enabled_modules.append('attack_vectors')
        if self.conversation_flow.enabled:
            enabled_modules.append('conversation_flow')
        if self.response_patterns.enabled:
            enabled_modules.append('response_patterns')
        if self.tool_calling.enabled:
            enabled_modules.append('tool_calling')
        if self.prompt_optimization.enabled:
            enabled_modules.append('prompt_optimization')
        return enabled_modules