"""
Tutorial and hint management system.
"""
from pathlib import Path
from typing import Optional
from src.backend.core.config import LETTER_GIFS_DIR


class TutorialManager:
    """Manages tutorial GIFs and hint system."""
    
    def __init__(self):
        """Initialize tutorial manager."""
        self.gifs_dir = LETTER_GIFS_DIR
        self.available_gifs = self._scan_available_gifs()
    
    def _scan_available_gifs(self) -> dict:
        """Scan for available letter GIF files."""
        available = {}
        if self.gifs_dir.exists():
            for gif_file in self.gifs_dir.glob("*.gif"):
                letter = gif_file.stem.upper()
                if len(letter) == 1 and letter.isalpha():
                    available[letter] = gif_file
        return available
    
    def has_tutorial(self, letter: Optional[str]) -> bool:
        """Check if tutorial GIF exists for letter."""
        if not letter:
            return False
        return letter.upper() in self.available_gifs
    
    def get_tutorial_path(self, letter: Optional[str]) -> Optional[Path]:
        """Get path to tutorial GIF for letter."""
        if not letter:
            return None
        return self.available_gifs.get(letter.upper())
    
    def get_tutorial_url(self, letter: Optional[str]) -> Optional[str]:
        """Get relative URL for tutorial GIF."""
        if not letter:
            return None
        if self.has_tutorial(letter):
            return f"/assets/{letter.upper()}.gif"
        return None
    
    def should_show_hint(self, attempt_count: int, hints_shown: int, max_hints: int) -> bool:
        """
        Determine if hint should be shown.
        
        Args:
            attempt_count: Number of failed attempts
            hints_shown: Number of hints already shown
            max_hints: Maximum hints allowed
            
        Returns:
            True if hint should be shown
        """
        if hints_shown >= max_hints:
            return False
        
        # Show first hint after 5 attempts, second after 10
        thresholds = [5, 10, 15]
        return attempt_count in thresholds
    
    def get_hint_message(self, letter: Optional[str], hint_number: int) -> str:
        """
        Get encouragement message for hint.
        
        Args:
            letter: Target letter
            hint_number: Which hint (1, 2, etc.)
            
        Returns:
            Hint message
        """
        if not letter:
            return ""
            
        messages = {
            1: f"Need help with '{letter}'? Check the example on the side.",
            2: f"Keep trying! Make sure your hand matches the '{letter}' shape exactly.",
            3: f"Almost there! Focus on the finger positions for '{letter}'."
        }
        return messages.get(hint_number, f"Keep practicing '{letter}'!")
