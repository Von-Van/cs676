#!/usr/bin/env python3
"""
persona_validator.py - Advanced Persona Validation and Consistency Checking

Provides comprehensive validation to ensure persona consistency and realism including:
- Behavioral consistency analysis
- Response pattern validation
- Character trait coherence checking
- Role-appropriate language verification
- Contradiction detection
- Realism scoring
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
import json


@dataclass
class ValidationResult:
    """Result of persona validation."""
    is_valid: bool
    consistency_score: float  # 0.0 to 1.0
    realism_score: float  # 0.0 to 1.0
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]
    metrics: Dict[str, Any]


class PersonaValidator:
    """Advanced validation for persona consistency and realism."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Trait compatibility matrix (positive and negative correlations)
        self.trait_incompatibilities = {
            'introverted': ['outgoing', 'extroverted', 'gregarious'],
            'extroverted': ['shy', 'introverted', 'reserved'],
            'analytical': ['impulsive', 'emotional', 'spontaneous'],
            'creative': ['rigid', 'inflexible', 'conservative'],
            'detail-oriented': ['big-picture', 'careless', 'hasty'],
            'empathetic': ['cold', 'uncaring', 'callous'],
            'decisive': ['indecisive', 'hesitant', 'uncertain'],
            'optimistic': ['pessimistic', 'cynical', 'negative'],
        }
        
        # Role-appropriate language patterns
        self.role_patterns = {
            'technical': ['implement', 'optimize', 'architecture', 'scale', 'performance'],
            'business': ['roi', 'stakeholder', 'market', 'revenue', 'strategy'],
            'design': ['user experience', 'interface', 'aesthetic', 'usability', 'visual'],
            'research': ['hypothesis', 'data', 'findings', 'methodology', 'evidence'],
            'leadership': ['vision', 'team', 'delegate', 'inspire', 'align'],
        }
        
        # Red flags for unrealistic personas
        self.red_flags = [
            'perfect', 'flawless', 'never wrong', 'always right',
            'knows everything', 'expert in everything', 'no weaknesses'
        ]
    
    def validate_persona_structure(self, persona: Dict[str, Any]) -> ValidationResult:
        """Comprehensive validation of persona structure and content."""
        issues = []
        warnings = []
        suggestions = []
        metrics = {}
        
        # 1. Required field validation
        required_fields = ['id', 'name', 'role', 'description', 'personality']
        for field in required_fields:
            if field not in persona or not persona[field]:
                issues.append(f"Missing required field: {field}")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                consistency_score=0.0,
                realism_score=0.0,
                issues=issues,
                warnings=warnings,
                suggestions=suggestions,
                metrics=metrics
            )
        
        # 2. Personality structure validation
        personality = persona.get('personality', {})
        if not isinstance(personality, dict):
            issues.append("Personality must be a dictionary")
        else:
            required_personality_fields = ['traits', 'goals', 'style']
            for field in required_personality_fields:
                if field not in personality:
                    warnings.append(f"Missing recommended personality field: {field}")
        
        # 3. Trait consistency check
        trait_score, trait_issues, trait_warnings = self._validate_trait_consistency(personality)
        issues.extend(trait_issues)
        warnings.extend(trait_warnings)
        metrics['trait_consistency_score'] = trait_score
        
        # 4. Role-description alignment check
        role_score, role_issues, role_suggestions = self._validate_role_alignment(
            persona.get('role', ''),
            persona.get('description', ''),
            personality
        )
        issues.extend(role_issues)
        suggestions.extend(role_suggestions)
        metrics['role_alignment_score'] = role_score
        
        # 5. Realism check
        realism_score, realism_warnings, realism_suggestions = self._validate_realism(persona)
        warnings.extend(realism_warnings)
        suggestions.extend(realism_suggestions)
        metrics['realism_score'] = realism_score
        
        # 6. Complexity and depth check
        depth_score, depth_suggestions = self._validate_depth(persona)
        suggestions.extend(depth_suggestions)
        metrics['depth_score'] = depth_score
        
        # 7. Language and tone consistency
        tone_score, tone_warnings = self._validate_tone_consistency(persona)
        warnings.extend(tone_warnings)
        metrics['tone_consistency_score'] = tone_score
        
        # Calculate overall scores
        consistency_score = (trait_score + role_score + tone_score) / 3
        overall_realism = (realism_score + depth_score) / 2
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            consistency_score=consistency_score,
            realism_score=overall_realism,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            metrics=metrics
        )
    
    def _validate_trait_consistency(self, personality: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Check for contradictory personality traits."""
        issues = []
        warnings = []
        
        traits = personality.get('traits', [])
        if isinstance(traits, str):
            traits = [t.strip() for t in traits.split(',')]
        
        if not traits:
            warnings.append("No personality traits defined")
            return 0.5, issues, warnings
        
        # Normalize traits to lowercase for comparison
        normalized_traits = [t.lower().strip() for t in traits]
        
        # Check for contradictions
        contradictions = 0
        total_checks = 0
        
        for i, trait in enumerate(normalized_traits):
            for incomp_base, incomp_list in self.trait_incompatibilities.items():
                if incomp_base in trait:
                    for other_trait in normalized_traits[i+1:]:
                        total_checks += 1
                        for incomp in incomp_list:
                            if incomp in other_trait:
                                contradictions += 1
                                issues.append(
                                    f"Contradictory traits detected: '{traits[i]}' conflicts with "
                                    f"'{normalized_traits[normalized_traits.index(other_trait)]}'"
                                )
        
        # Check for duplicate traits
        trait_counts = Counter(normalized_traits)
        for trait, count in trait_counts.items():
            if count > 1:
                warnings.append(f"Duplicate trait: '{trait}' appears {count} times")
        
        # Calculate consistency score
        if total_checks == 0:
            consistency_score = 0.8  # No obvious conflicts checked
        else:
            consistency_score = 1.0 - (contradictions / max(total_checks, 1))
        
        return consistency_score, issues, warnings
    
    def _validate_role_alignment(self, role: str, description: str, personality: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Check if personality aligns with stated role."""
        issues = []
        suggestions = []
        
        role_lower = role.lower()
        desc_lower = description.lower()
        combined_text = f"{role_lower} {desc_lower}"
        
        # Identify role category
        role_category = None
        for category, keywords in self.role_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                role_category = category
                break
        
        if not role_category:
            suggestions.append(
                f"Consider adding more role-specific details for '{role}' to improve clarity"
            )
            return 0.7, issues, suggestions
        
        # Check if personality traits support the role
        traits = personality.get('traits', [])
        if isinstance(traits, str):
            traits = [t.strip() for t in traits.split(',')]
        
        traits_text = ' '.join([str(t).lower() for t in traits])
        
        # Role-specific trait expectations
        expected_traits = {
            'technical': ['analytical', 'logical', 'detail-oriented', 'systematic'],
            'business': ['strategic', 'results-oriented', 'persuasive', 'analytical'],
            'design': ['creative', 'empathetic', 'visual', 'user-focused'],
            'research': ['curious', 'methodical', 'analytical', 'thorough'],
            'leadership': ['decisive', 'inspirational', 'strategic', 'communicative'],
        }
        
        if role_category in expected_traits:
            matching_traits = sum(
                1 for exp in expected_traits[role_category]
                if exp in traits_text
            )
            
            if matching_traits == 0:
                suggestions.append(
                    f"Consider adding traits typical for {role_category} roles: "
                    f"{', '.join(expected_traits[role_category])}"
                )
                alignment_score = 0.5
            else:
                alignment_score = min(1.0, 0.5 + (matching_traits * 0.15))
        else:
            alignment_score = 0.7
        
        return alignment_score, issues, suggestions
    
    def _validate_realism(self, persona: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
        """Check for unrealistic or overly perfect personas."""
        warnings = []
        suggestions = []
        
        description = persona.get('description', '').lower()
        personality = persona.get('personality', {})
        
        # Check for red flag phrases
        red_flag_count = sum(1 for flag in self.red_flags if flag in description)
        
        if red_flag_count > 0:
            warnings.append(
                "Persona may be unrealistic - avoid 'perfect' or 'flawless' characterizations"
            )
        
        # Check for weaknesses or limitations
        has_weaknesses = any(
            word in description
            for word in ['weakness', 'struggle', 'challenge', 'difficulty', 'limitation']
        )
        
        if not has_weaknesses:
            suggestions.append(
                "Consider adding realistic weaknesses or challenges to make the persona more believable"
            )
        
        # Check personality depth
        traits = personality.get('traits', [])
        if isinstance(traits, str):
            traits = [t.strip() for t in traits.split(',')]
        
        if len(traits) < 3:
            suggestions.append(
                "Add more personality traits (at least 3-5) for a well-rounded character"
            )
        elif len(traits) > 10:
            warnings.append(
                "Too many traits may make the persona unfocused - consider narrowing to 5-7 key traits"
            )
        
        # Calculate realism score
        realism_score = 1.0
        realism_score -= (red_flag_count * 0.2)  # Penalize red flags
        if not has_weaknesses:
            realism_score -= 0.1
        if len(traits) < 3:
            realism_score -= 0.15
        elif len(traits) > 10:
            realism_score -= 0.05
        
        return max(0.0, min(1.0, realism_score)), warnings, suggestions
    
    def _validate_depth(self, persona: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check for sufficient persona depth and complexity."""
        suggestions = []
        
        description = persona.get('description', '')
        personality = persona.get('personality', {})
        
        # Check description length
        desc_length = len(description.split())
        if desc_length < 20:
            suggestions.append(
                "Expand description to at least 20-30 words for better context"
            )
        
        # Check for goals
        goals = personality.get('goals', [])
        if not goals:
            suggestions.append("Add specific goals or motivations for the persona")
        elif isinstance(goals, list) and len(goals) < 2:
            suggestions.append("Add multiple goals (2-4) to create complexity")
        
        # Check for background/context
        has_background = any(
            word in description.lower()
            for word in ['background', 'experience', 'worked', 'studied', 'years']
        )
        
        if not has_background:
            suggestions.append(
                "Include background or experience details to add authenticity"
            )
        
        # Calculate depth score
        depth_score = 0.5  # Base score
        
        if desc_length >= 20:
            depth_score += 0.2
        if goals and (isinstance(goals, list) and len(goals) >= 2):
            depth_score += 0.15
        if has_background:
            depth_score += 0.15
        
        return min(1.0, depth_score), suggestions
    
    def _validate_tone_consistency(self, persona: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Check for consistent tone and style."""
        warnings = []
        
        personality = persona.get('personality', {})
        style = personality.get('style', '')
        
        if not style:
            warnings.append("No communication style defined - add 'style' to personality")
            return 0.6, warnings
        
        style_lower = style.lower()
        
        # Check for contradictory style indicators
        formal_indicators = ['formal', 'professional', 'corporate', 'structured']
        casual_indicators = ['casual', 'informal', 'relaxed', 'conversational']
        
        has_formal = any(ind in style_lower for ind in formal_indicators)
        has_casual = any(ind in style_lower for ind in casual_indicators)
        
        if has_formal and has_casual:
            warnings.append(
                "Communication style contains contradictory elements (both formal and casual)"
            )
            return 0.5, warnings
        
        return 1.0, warnings
    
    def validate_conversation_consistency(self, persona: Dict[str, Any], messages: List[str]) -> Dict[str, Any]:
        """Validate that conversation messages align with persona characteristics."""
        
        personality = persona.get('personality', {})
        traits = personality.get('traits', [])
        if isinstance(traits, str):
            traits = [t.strip().lower() for t in traits.split(',')]
        
        style = personality.get('style', '').lower()
        
        inconsistencies = []
        metrics = {
            'total_messages': len(messages),
            'avg_message_length': sum(len(m.split()) for m in messages) / len(messages) if messages else 0,
        }
        
        # Check style consistency
        if 'formal' in style:
            casual_words = ['gonna', 'wanna', 'yeah', 'nah', 'hey']
            for i, msg in enumerate(messages):
                msg_lower = msg.lower()
                for word in casual_words:
                    if word in msg_lower:
                        inconsistencies.append(
                            f"Message {i+1}: Casual language ('{word}') inconsistent with formal style"
                        )
        
        # Check trait manifestation
        if 'analytical' in ' '.join(traits):
            analytical_indicators = ['data', 'analysis', 'consider', 'examine', 'evaluate']
            has_analytical = any(
                any(ind in msg.lower() for ind in analytical_indicators)
                for msg in messages
            )
            if not has_analytical:
                inconsistencies.append(
                    "Analytical trait not reflected in conversation - consider using more analytical language"
                )
        
        return {
            'is_consistent': len(inconsistencies) == 0,
            'inconsistencies': inconsistencies,
            'metrics': metrics
        }
    
    def generate_improvement_suggestions(self, validation_result: ValidationResult) -> List[str]:
        """Generate actionable improvement suggestions based on validation results."""
        improvements = []
        
        if validation_result.consistency_score < 0.7:
            improvements.append(
                "⚠️ LOW CONSISTENCY: Review personality traits for contradictions. "
                "Ensure traits complement rather than contradict each other."
            )
        
        if validation_result.realism_score < 0.7:
            improvements.append(
                "⚠️ LOW REALISM: Add weaknesses, challenges, or limitations. "
                "Avoid 'perfect' characterizations - realistic personas have flaws."
            )
        
        if validation_result.issues:
            improvements.append(
                f"🔴 CRITICAL ISSUES ({len(validation_result.issues)}): "
                "Fix required field issues before proceeding."
            )
        
        if validation_result.warnings:
            improvements.append(
                f"🟡 WARNINGS ({len(validation_result.warnings)}): "
                "Address warnings to improve persona quality."
            )
        
        # Add top 3 suggestions
        if validation_result.suggestions:
            improvements.append("💡 TOP SUGGESTIONS:")
            for suggestion in validation_result.suggestions[:3]:
                improvements.append(f"  - {suggestion}")
        
        return improvements


def validate_persona_batch(personas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a batch of personas and provide summary statistics."""
    validator = PersonaValidator()
    
    results = []
    for persona in personas:
        result = validator.validate_persona_structure(persona)
        results.append({
            'persona_name': persona.get('name', 'Unknown'),
            'result': result
        })
    
    # Calculate aggregate statistics
    valid_count = sum(1 for r in results if r['result'].is_valid)
    avg_consistency = sum(r['result'].consistency_score for r in results) / len(results) if results else 0
    avg_realism = sum(r['result'].realism_score for r in results) / len(results) if results else 0
    
    return {
        'total_personas': len(personas),
        'valid_personas': valid_count,
        'invalid_personas': len(personas) - valid_count,
        'average_consistency_score': avg_consistency,
        'average_realism_score': avg_realism,
        'individual_results': results
    }
