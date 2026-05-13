/**
 * Utility functions for processing server responses in the Sign Language Learning App
 */

export type ServerMessage = Record<string, unknown>;

export interface NormalizedProgress {
  correct: number;
  attempts: number;
  accuracy: string;
  timeRemaining: string;
}

export interface NormalizedPrediction {
  displayLetter: string;
  confidence: number;
}

export interface DotsState {
  current: number;
  required: number;
  isMatch: boolean;
}

/**
 * Extract and normalize progress statistics from server response
 */
export function normalizeProgress(data: ServerMessage): NormalizedProgress | null {
  const correct = typeof data.correct === 'number' ? data.correct : undefined;
  const attempts = typeof data.attempts === 'number' ? data.attempts : undefined;
  const accuracy = typeof data.accuracy === 'string' ? data.accuracy : undefined;
  const timeRemaining = typeof data.time_remaining === 'string' ? data.time_remaining : undefined;

  if (correct !== undefined && attempts !== undefined) {
    return {
      correct,
      attempts,
      accuracy: accuracy ?? '0.0',
      timeRemaining: timeRemaining ?? ''
    };
  }

  return null;
}

/**
 * Extract target letter from server response
 */
export function getTargetLetter(data: ServerMessage): string | null {
  const targetLetter = data.target_letter;
  return typeof targetLetter === 'string' ? targetLetter : null;
}

/**
 * Extract tutorial URL from server response
 */
export function getTutorialUrl(data: ServerMessage): string | null | undefined {
  const tutorialUrl = data.tutorial_url;
  if (typeof tutorialUrl === 'string') {
    return tutorialUrl;
  }
  if (tutorialUrl === null) {
    return null;
  }
  return undefined;
}

/**
 * Extract and normalize prediction from server response
 */
export function normalizePrediction(data: ServerMessage): NormalizedPrediction | null {
  if (typeof data.prediction !== 'object' || data.prediction === null) {
    return null;
  }

  const prediction = data.prediction as Record<string, unknown>;
  const letter = prediction.letter;
  const confidence = prediction.confidence;

  if (typeof letter === 'string' && typeof confidence === 'number') {
    return {
      displayLetter: letter,
      confidence: confidence
    };
  }

  return null;
}

/**
 * Extract recognized sentence from server response
 */
export function extractRecognizedSentence(data: ServerMessage): string | null {
  const sentence = data.recognized_sentence;
  if (typeof sentence === 'string') {
    return sentence;
  }
  return typeof sentence === 'undefined' ? null : undefined as any;
}

/**
 * Advance dots state based on prediction and target letter
 */
export function advanceDots(
  currentDots: DotsState,
  targetLetter: string | null,
  prediction: NormalizedPrediction
): DotsState {
  if (!targetLetter) {
    return currentDots;
  }

  const isMatch = prediction.displayLetter.toUpperCase() === targetLetter.toUpperCase();

  if (isMatch) {
    return {
      current: currentDots.current + 1,
      required: currentDots.required,
      isMatch: true
    };
  }

  return {
    ...currentDots,
    isMatch: false
  };
}
