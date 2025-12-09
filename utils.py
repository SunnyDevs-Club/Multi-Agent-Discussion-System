from pathlib import Path
from typing import Any, Dict, List, Union
import yaml


def get_agent_system_prompt(file_path: Path | str) -> str:
    """
    Retrieves and formats the system prompt for a given agent from a YAML file.
    Handles diverse YAML structures flexibly.

    Args:
        file_path: Path to the YAML file containing the agent configuration.

    Returns:
        A single, structured string containing the combined system prompt.
    
    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    if not file_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {file_path}")

    # Load the YAML content safely
    with open(file_path, 'r', encoding='utf-8') as file:
        data: Dict[str, Any] = yaml.safe_load(file)

    # Build prompt dynamically based on available fields
    prompt_parts = []
    
    # Process each key in the YAML flexibly
    for key, value in data.items():
        if value is None or value == "":
            continue
            
        formatted_section = _format_section(key, value)
        if formatted_section:
            prompt_parts.append(formatted_section)

    # Combine all parts
    if prompt_parts:
        final_prompt = (
            "### SYSTEM INSTRUCTION FILE START ###\n\n"
            "Always answer without any markdown formatting and do not include any tables or lists unless explicitly instructed to do so. "
            "Since it is a dialogue keep your answers short, concise, and to the point. Your response should not exceed 2 paragraphs in length or 1 minute of speech with normal pace. "
            "Always keep in mind that this is a dialogue between two AI models representing personas of specific professors whose persona is specified below. Both professors are arguing/debating on a topic provided by a user, who is a student seeking to understand different perspectives on the topic. "
            "You should NEVER break a character provided to you. Always think and response critically and analytically, but all your critique MUST BE based on facts and logical reasoning.\n\n"
            "Read and strictly adhere to the following configuration which provides contextual information and constraints for your entire session.\n\n"
            "```command_config\n"
            + "\n\n".join(prompt_parts) + 
            "\n```\n\n"
            "AND ONCE MORE, NO MARKDOWN, NO TABLES, ONLY DIALOGUE STYLE. SHOULD NOT EXCEED 200 CHARS\n"
            "### SYSTEM INSTRUCTION FILE END ###"
        )
    else:
        final_prompt = "### SYSTEM INSTRUCTION FILE START ###\n\nNo configuration provided.\n\n### SYSTEM INSTRUCTION FILE END ###"
    
    return final_prompt.strip()


def _format_section(key: str, value: Any) -> str:
    """
    Formats a YAML section based on its key and value type.
    
    Args:
        key: The section name from YAML
        value: The section content (can be string, list, dict, etc.)
    
    Returns:
        Formatted section string with appropriate tags
    """
    # Normalize the key to uppercase with underscores for tag names
    tag_name = key.upper().replace(" ", "_").replace("-", "_")
    
    # Handle different value types
    if isinstance(value, str):
        return f"<{tag_name}>\n{value.strip()}\n</{tag_name}>"
    
    elif isinstance(value, list):
        # Format lists as bullet points
        items = "\n".join(f"- {str(item).strip()}" for item in value if item)
        if items:
            return f"<{tag_name}>\n{items}\n</{tag_name}>"
    
    elif isinstance(value, dict):
        # Handle nested dictionaries
        return _format_dict_section(tag_name, value)
    
    elif isinstance(value, (int, float, bool)):
        return f"<{tag_name}>\n{value}\n</{tag_name}>"
    
    return ""


def _format_dict_section(tag_name: str, data: Dict[str, Any]) -> str:
    """
    Formats a dictionary section with special handling for common patterns.
    
    Args:
        tag_name: The tag name for this section
        data: Dictionary data to format
    
    Returns:
        Formatted section string
    """
    content_parts = []
    
    for sub_key, sub_value in data.items():
        if sub_value is None or sub_value == "":
            continue
        
        # Clean up the sub-key for display
        display_key = sub_key.replace("_", " ").title()
        
        if isinstance(sub_value, str):
            content_parts.append(f"{display_key}: {sub_value.strip()}")
        
        elif isinstance(sub_value, list):
            items = "; ".join(str(item).strip() for item in sub_value if item)
            if items:
                content_parts.append(f"{display_key}: {items}")
        
        elif isinstance(sub_value, dict):
            # Nested dict - format recursively
            nested_items = []
            for k, v in sub_value.items():
                if v:
                    nested_items.append(f"  {k}: {v}")
            if nested_items:
                content_parts.append(f"{display_key}:\n" + "\n".join(nested_items))
        
        else:
            content_parts.append(f"{display_key}: {sub_value}")
    
    if content_parts:
        content = "\n".join(content_parts)
        return f"<{tag_name}>\n{content}\n</{tag_name}>"
    
    return ""


# Example usage and testing function
def test_parser_with_your_yaml():
    """Test the parser with your actual YAML structure"""
    
    your_yaml = """
persona:
  name: "Prof. Ashish Seth"
  archetype: "Calm, wise, slow, methodical academic lecturer."
  description: >
    A thoughtful, slow-speaking, wise lecturer who explains concepts step by step.
    Maintains a gentle, academic manner and guides students through difficult
    topics calmly, patiently, and clearly.

  greeting_style: >
    Begins sessions with:
    "Hello everyone, today we are going to…"

  signature_phrases:
    - "Yes, yes, yes."
    - "Do you understand, or I need to repeat?"

core_mannerisms:
  - "Always speak slowly, with deliberate pauses."
  - "Use subtle academic phrasing without caricature."
  - "Maintain calm body language and gentle tone."

speech_patterns:
  pacing:
    - Uses soft fillers such as "so…", "uh…", "okay…" to maintain slow pacing.
    - Smooth transitions between steps.

  transitions:
    - "so let us quickly see…"
    - "now if you look…"
    - "let us recall…"
    - "let us look to this case…"

  storytelling:
    - Explains through calm, logical, step-by-step reasoning.
    - Uses mental structural diagrams and careful sequencing.

  humor:
    - Extremely subtle, respectful, minimal.

goals:
  - Provide clear, structured explanations like a calm lecturer.
  - Maintain slow, steady, wise energy.
  - Build deep understanding through step-by-step reasoning.
  - Encourage reflection rather than emotional excitement.

rules:
  - Maintain a strict, calm, academic tone at all times.
  - Avoid dangerous, illegal, or harmful instructions.
  - Do not imitate accents or stereotypes.
  - If a request is unsafe, ask for academic purpose before proceeding.
  - Do not break calm pacing.

style:
  tone: "Calm, wise, slow, thoughtful."
  structure:
    - Begin with a short intuitive foundation.
    - Explain step by step in a methodical way.
    - Use guiding phrases to direct attention.
    - Conclude with a concise summary.

  embedded_phrases:
    - "If you look to this structure…"
    - "You can see here…"
    - "Let us quickly cross-check…"
    - "We are into this case…"
    - "Let us draw the structure…"
    - "So this is your resultant structure."
    - "Now we are done with…"
    - "Thank you very much."

memory_policy: |
  Treat past messages as academic continuity only.
  Do not retain personal information beyond the session.
  Maintain short-term contextual consistency only.
"""
    
    # Parse the YAML
    import tempfile
    import os
    
    # Create temp file and ensure it's properly written
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(your_yaml)
        temp_path = f.name
    
    try:
        result = get_agent_system_prompt(temp_path)
        print("Generated System Prompt:")
        print("=" * 70)
        print(result)
        print("=" * 70)
        print("\n✓ Parser successfully handled your nested YAML structure!")
    finally:
        os.unlink(temp_path)
    

if __name__ == "__main__":
    test_parser_with_your_yaml()
