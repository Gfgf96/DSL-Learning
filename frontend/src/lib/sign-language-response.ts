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
  const progress = (data.progress as Record<string, unknown>) || {};
  
  const correctRaw = progress.total_correct ?? data.total_correct ?? data.correct;
  const attemptsRaw = progress.total_attempts ?? data.total_attempts ?? data.attempts;
  const accuracyRaw = progress.accuracy ?? data.accuracy;
  const timeRemainingRaw = progress.time_remaining ?? data.time_remaining;

  const correct = typeof correctRaw === 'number' ? correctRaw : undefined;
  const attempts = typeof attemptsRaw === 'number' ? attemptsRaw : undefined;
  const accuracy = typeof accuracyRaw === 'number' ? accuracyRaw.toFixed(1) : (typeof accuracyRaw === 'string' ? accuracyRaw : undefined);
  const timeRemaining = typeof timeRemainingRaw === 'number' ? timeRemainingRaw.toFixed(1) : (typeof timeRemainingRaw === 'string' ? timeRemainingRaw : undefined);

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
  const progress = (data.progress as Record<string, unknown>) || {};
  const targetLetter = data.target_letter ?? progress.current_letter ?? data.current_letter;
  return typeof targetLetter === 'string' ? targetLetter : null;
}

/**
 * Extract tutorial URL from server response
 */
export function getTutorialUrl(data: ServerMessage): string | null | undefined {
  const progress = (data.progress as Record<string, unknown>) || {};
  const tutorialUrl = data.tutorial_url ?? progress.tutorial_url;
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
  const letter = prediction.letter ?? prediction.predicted_class;
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
  const progress = (data.progress as Record<string, unknown>) || {};
  const sentence = data.recognized_sentence ?? progress.recognized_sentence;
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
