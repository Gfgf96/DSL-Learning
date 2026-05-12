"""
Letter sequencing and curriculum management.
"""
import random
from typing import List, Optional
from src.backend.core.config import STATIC_LETTERS, DYNAMIC_LETTERS, ALL_LETTERS


class LetterSequence:
    """Manages the sequence of letters for training."""
    
    def __init__(self, mode: str = "sequential", include_dynamic: bool = False):
        """
        Initialize letter sequence.
        
        Args:
            mode: "sequential", "random", or "custom"
            include_dynamic: Whether to include J and Z (dynamic letters)
        """
        self.mode = mode
        self.include_dynamic = include_dynamic
        self.available_letters = (
            ALL_LETTERS if include_dynamic else STATIC_LETTERS
        )
        self.current_index = 0
        self.completed_letters = []
        
    def get_next_letter(self, current_letter: Optional[str] = None) -> str:
        """
        Get the next letter in the sequence.
        
        Args:
            current_letter: Current letter (to avoid repeating)
            
        Returns:
            Next letter to practice
        """
        if self.mode == "sequential":
            return self._get_sequential(current_letter)
        elif self.mode == "random":
            return self._get_random(current_letter)
        else:
            return self._get_sequential(current_letter)
    
    def _get_sequential(self, current_letter: Optional[str]) -> str:
        """Get next letter in alphabetical order."""
        if current_letter and current_letter in self.available_letters:
            idx = self.available_letters.index(current_letter)
            self.current_index = (idx + 1) % len(self.available_letters)
        
        letter = self.available_letters[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.available_letters)
        return letter
    
    def _get_random(self, current_letter: Optional[str]) -> str:
        """Get random letter (avoiding current)."""
        candidates = [l for l in self.available_letters if l != current_letter]
        if not candidates:
            candidates = self.available_letters
        choice = random.choice(candidates)
        print(f"🎲 Random mode: selected '{choice}' from {len(candidates)} candidates (avoiding '{current_letter}')")
        return choice
    
    def mark_completed(self, letter: str):
        """Mark a letter as successfully completed."""
        if letter not in self.completed_letters:
            self.completed_letters.append(letter)
    
    def get_progress(self) -> dict:
        """Get current progress statistics."""
        total = len(self.available_letters)
        completed = len(self.completed_letters)
        return {
            "total": total,
            "completed": completed,
            "percentage": (completed / total * 100) if total > 0 else 0,
            "completed_letters": self.completed_letters
        }
    
    def reset(self):
        """Reset progress."""
        self.current_index = 0
        self.completed_letters = []
