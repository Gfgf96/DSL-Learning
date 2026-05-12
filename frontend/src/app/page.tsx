"use client";

import React, { useEffect, useState, useRef, useCallback } from 'react';
import Image from 'next/image';
import { SkipForward, Play, Square } from 'lucide-react';
import { ThemeToggle } from '@/components/ThemeToggle';
import { WebcamView } from '@/components/WebcamView';
import {
  advanceDots,
  extractRecognizedSentence,
  getTargetLetter,
  getTutorialUrl,
  normalizePrediction,
  normalizeProgress
} from '@/lib/sign-language-response';

type ServerMessage = Record<string, unknown>;

const getStringField = (value: unknown): string | null => (typeof value === 'string' ? value : null);
const getBooleanField = (value: unknown): boolean | undefined => (typeof value === 'boolean' ? value : undefined);
const getNumberField = (value: unknown): number | undefined => (typeof value === 'number' ? value : undefined);

export default function Home() {
  // WebSocket Refs
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  // App State
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isWaitingForServer, setIsWaitingForServer] = useState(false);
  
  // Settings State
  const [practiceMode, setPracticeMode] = useState<'letter' | 'sentence' | 'free'>('letter');
  const [letterOrder, setLetterOrder] = useState<'sequential' | 'random'>('sequential');
  const [language, setLanguage] = useState<'en' | 'nl'>('en');
  
  // UI State
  const [status, setStatus] = useState({ text: 'Not Connected', connected: false, handDetected: false });
  const [target, setTarget] = useState({ letter: 'A', hint: '', image: '' });
  const [prediction, setPrediction] = useState({ letter: '-', confidence: 0 });
  const [stats, setStats] = useState({ correct: 0, attempts: 0, accuracy: '0.0', timeRemaining: '' });
  const [progress, setProgress] = useState(0);
  const [dots, setDots] = useState({ current: 0, required: 0, isMatch: false });
  
  // Sentence State
  const [targetSentenceInput, setTargetSentenceInput] = useState('');
  const [activeTargetSentence, setActiveTargetSentence] = useState('');
  const [recognizedSentence, setRecognizedSentence] = useState<Array<{char: string, correct: boolean}>>([]);
  
  const [toastMessage, setToastMessage] = useState<{type: 'hint'|'success'|'error', text: string} | null>(null);

  // Translations
  const t = {
    en: {
      title: 'Sign Language Learning',
      subtitle: 'Learn NGT alphabet with real-time feedback',
      notConnected: 'Not Connected',
      connected: 'Connected',
      handDetected: 'Hand Detected',
      recording: 'Recording',
      startCamera: 'Start Camera',
      record: 'Record',
      stopRecording: 'Stop Recording',
      skipLetter: 'Skip Letter',
      targetLetter: 'Target Letter',
      prediction: 'Prediction',
      correct: 'Correct',
      attempts: 'Attempts',
      accuracy: 'Accuracy',
      timeRemaining: 'Time Remaining',
      practiceMode: 'Practice Mode',
      letterOrder: 'Letter Order',
      language: 'Language'
    },
    nl: {
      title: 'Gebarentaal Leren',
      subtitle: 'Leer het NGT alfabet met real-time feedback',
      notConnected: 'Niet verbonden',
      connected: 'Verbonden',
      handDetected: 'Hand gedetecteerd',
      recording: 'Opnemen',
      startCamera: 'Start Camera',
      record: 'Neem op',
      stopRecording: 'Stop Opname',
      skipLetter: 'Sla over',
      targetLetter: 'Doelletter',
      prediction: 'Voorspelling',
      correct: 'Goed',
      attempts: 'Pogingen',
      accuracy: 'Nauwkeurigheid',
      timeRemaining: 'Resterende Tijd',
      practiceMode: 'Oefenmodus',
      letterOrder: 'Lettervolgorde',
      language: 'Taal'
    }
  };

  const currentT = t[language];

  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }

      wsRef.current?.close();
    };
  }, []);

  const showToast = (type: 'hint'|'success'|'error', text: string) => {
    setToastMessage({ type, text });
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleServerResponse = useCallback((data: ServerMessage) => {
    setIsWaitingForServer(false);

    const incomingSessionId = getStringField(data.session_id);
    if (incomingSessionId && !sessionId) {
      setSessionId(incomingSessionId);
    }

    const errorMessage = getStringField(data.error);
    const responseMessage = getStringField(data.message);
    const recordingState = getBooleanField(data.recording);
    const handDetectedState = getBooleanField(data.hand_detected) ?? false;
    const bufferProgress = getNumberField(data.buffer_progress);

    if (errorMessage) {
      console.error(errorMessage);
      if (responseMessage === "Connection timeout") {
        setStatus(s => ({ ...s, connected: false, text: currentT.notConnected }));
      }
      return;
    }

    // Status updates
    setStatus({
      connected: true,
      handDetected: handDetectedState,
      text: recordingState ? currentT.recording : (handDetectedState ? currentT.handDetected : currentT.connected)
    });

    if (recordingState !== undefined) {
      setIsRecording(recordingState);
    }

    if (bufferProgress !== undefined) {
      setProgress(bufferProgress * 100 || 0);
    } else if (recordingState === false) {
      setProgress(0);
    }

    // Process result messages
    if (responseMessage) {
      if (responseMessage.includes('Success')) {
        showToast('success', responseMessage);
      } else if (responseMessage.includes('Failed attempt')) {
        showToast('error', responseMessage);
      } else if (responseMessage.includes('Recording in progress') || responseMessage.includes('Started recording')) {
        // Just status updates
      } else {
        showToast('hint', responseMessage);
      }
    }

    const normalizedProgress = normalizeProgress(data);
    if (normalizedProgress) {
      setStats(normalizedProgress);
    }

    const targetLetter = getTargetLetter(data);
    if (targetLetter) {
      setTarget(t => ({ ...t, letter: targetLetter }));
    }

    const tutorialUrl = getTutorialUrl(data);
    if (tutorialUrl) {
      setTarget(t => ({ ...t, image: tutorialUrl }));
    } else if (tutorialUrl === null) {
      setTarget(t => ({ ...t, image: '' }));
    }

    // Prediction normalization
    const normalizedPrediction = normalizePrediction(data);
    if (normalizedPrediction) {
      setPrediction({ letter: normalizedPrediction.displayLetter, confidence: normalizedPrediction.confidence });
    }

    // Dots logic based on normalized prediction and current target letter
    if (normalizedPrediction) {
      setDots(previous => advanceDots(previous, targetLetter, normalizedPrediction));
    }

    // Sentence mode recognized sentence
    const recSentence = extractRecognizedSentence(data);
    if (recSentence !== null) {
      setActiveTargetSentence(targetStr => {
        const formatted = Array.from(recSentence).map((char: string, i: number) => ({
          char,
          correct: targetStr && i < targetStr.length ? char.toUpperCase() === targetStr[i].toUpperCase() : true
        }));
        setRecognizedSentence(formatted);
        return targetStr;
      });
    }

  }, [sessionId, currentT]);

  // WebSocket Connection
  function connectWebSocket(activeSessionId?: string | null) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Use window.location.host, or fallback to localhost:8000 for development
    const host = process.env.NODE_ENV === 'development' ? 'localhost:8000' : window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;
    
    console.log(`Connecting to ${wsUrl}...`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setStatus(s => ({ ...s, connected: true, text: currentT.connected }));
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      const sessionToResume = activeSessionId ?? sessionIdRef.current;
      if (sessionToResume) {
        ws.send(JSON.stringify({ type: 'reconnect', session_id: sessionToResume }));
      }
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleServerResponse(data);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setStatus(s => ({ ...s, connected: false, text: currentT.notConnected }));
      setIsRecording(false);
      
      // Auto reconnect
      reconnectTimeoutRef.current = setTimeout(() => {
        if (isCameraActive) connectWebSocket(sessionIdRef.current);
      }, 3000);
    };

    wsRef.current = ws;
  }

  const handleStartCamera = () => {
    setIsCameraActive(true);
    // Create a session on the server so UI controls can update immediately
    createSession().then((newSessionId) => {
      connectWebSocket(newSessionId);
    }).catch(() => {
      // Still attempt to connect even if session creation failed
      connectWebSocket();
    });
  };

  const createSession = async (): Promise<string | null> => {
    try {
      const host = process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '';
      const resp = await fetch(`${host}/api/session/new`, { method: 'POST' });
      if (resp.ok) {
        const data = await resp.json();
        if (data.session_id) {
          sessionIdRef.current = data.session_id;
          setSessionId(data.session_id);
          return data.session_id;
        }
      } else {
        console.warn('Failed to create session on server');
      }
    } catch (e) {
      console.error('createSession error', e);
    }

    return null;
  };

  const handleFrameCapture = useCallback((frameData: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && !isWaitingForServer) {
      setIsWaitingForServer(true);
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        frame: frameData,
        session_id: sessionIdRef.current
      }));
    }
  }, [isWaitingForServer]);

  // Commands
  const toggleRecording = () => {
    const activeSessionId = sessionIdRef.current;
    if (!wsRef.current || !activeSessionId) return;
    const type = isRecording ? 'stop_recording' : 'start_recording';
    wsRef.current.send(JSON.stringify({ type, session_id: activeSessionId }));
  };

  const skipLetter = () => {
    const activeSessionId = sessionIdRef.current;
    if (!wsRef.current || !activeSessionId) return;
    wsRef.current.send(JSON.stringify({ type: 'skip', session_id: activeSessionId }));
  };

  const updateMode = async (newMode: string) => {
    const activeSessionId = sessionIdRef.current;
    if (!activeSessionId) return;
    try {
      const host = process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '';
      await fetch(`${host}/api/session/${activeSessionId}/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: newMode })
      });
    } catch (e) {
      console.error(e);
    }
  };

  const handlePracticeModeChange = (mode: 'letter' | 'sentence' | 'free') => {
    setPracticeMode(mode);
    const actualMode = mode === 'letter' ? letterOrder : (mode === 'free' ? 'sentence' : mode);
    updateMode(actualMode);
    
    const activeSessionId = sessionIdRef.current;
    if (mode === 'free' && wsRef.current && activeSessionId) {
      wsRef.current.send(JSON.stringify({ type: 'set_sentence', sentence: '', session_id: activeSessionId }));
      setActiveTargetSentence('');
    }
  };

  const handleOrderChange = (order: 'sequential' | 'random') => {
    setLetterOrder(order);
    if (practiceMode === 'letter') {
      updateMode(order);
    }
  };

  const applyTargetSentence = () => {
    const activeSessionId = sessionIdRef.current;
    if (!wsRef.current || !activeSessionId) return;
    setActiveTargetSentence(targetSentenceInput);
    wsRef.current.send(JSON.stringify({ 
      type: 'set_sentence', 
      sentence: targetSentenceInput, 
      session_id: activeSessionId 
    }));
  };

  const clearSentence = () => {
    const activeSessionId = sessionIdRef.current;
    if (!wsRef.current || !activeSessionId) return;
    wsRef.current.send(JSON.stringify({ type: 'clear_sentence', session_id: activeSessionId }));
    setRecognizedSentence([]);
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Toast Notification */}
      {toastMessage && (
        <div className={`fixed bottom-8 left-1/2 -translate-x-1/2 px-6 py-3 rounded-xl shadow-xl z-50 transition-all ${
          toastMessage.type === 'success' ? 'bg-emerald-500 text-white' :
          toastMessage.type === 'error' ? 'bg-red-500 text-white' :
          'bg-white dark:bg-neutral-800 text-neutral-900 dark:text-white border border-neutral-200 dark:border-neutral-700'
        }`}>
          {toastMessage.text}
        </div>
      )}

      {/* Header */}
      <header className="border-b border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950 p-6 sticky top-0 z-10">
        <div className="max-w-[2400px] w-full px-4 md:px-8 2xl:px-16 mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="text-center md:text-left">
            <h1 className="text-3xl font-bold tracking-tight">{currentT.title}</h1>
            <p className="text-neutral-500 dark:text-neutral-400 mt-1">{currentT.subtitle}</p>
          </div>
          
          <div className="flex flex-wrap items-center justify-center gap-4">
            <ThemeToggle />

            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-semibold text-neutral-400 uppercase tracking-wider">{currentT.practiceMode}</span>
              <select 
                disabled={!isCameraActive}
                value={practiceMode}
                onChange={(e) => handlePracticeModeChange(e.target.value as 'letter' | 'sentence' | 'free')}
                className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-lg px-3 py-1.5 text-sm outline-none focus:ring-2 focus:ring-neutral-500 disabled:opacity-50"
              >
                <option value="letter">Single Letters</option>
                <option value="sentence">Target Sentence</option>
                <option value="free">Free Sign</option>
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-semibold text-neutral-400 uppercase tracking-wider">{currentT.letterOrder}</span>
              <div className="flex bg-neutral-100 dark:bg-neutral-900 rounded-lg p-1 border border-neutral-200 dark:border-neutral-800">
                <button 
                  disabled={!isCameraActive || practiceMode !== 'letter'}
                  onClick={() => handleOrderChange('sequential')}
                  className={`px-3 py-1 text-sm rounded-md transition-colors disabled:opacity-50 ${letterOrder === 'sequential' ? 'bg-white dark:bg-neutral-800 shadow-sm text-black dark:text-white' : 'text-neutral-500'}`}
                >ABC</button>
                <button 
                  disabled={!isCameraActive || practiceMode !== 'letter'}
                  onClick={() => handleOrderChange('random')}
                  className={`px-3 py-1 text-sm rounded-md transition-colors disabled:opacity-50 ${letterOrder === 'random' ? 'bg-white dark:bg-neutral-800 shadow-sm text-black dark:text-white' : 'text-neutral-500'}`}
                >Random</button>
              </div>
            </div>

            <div className="flex flex-col gap-1">
              <span className="text-[10px] font-semibold text-neutral-400 uppercase tracking-wider">{currentT.language}</span>
              <div className="flex bg-neutral-100 dark:bg-neutral-900 rounded-lg p-1 border border-neutral-200 dark:border-neutral-800">
                <button onClick={() => setLanguage('en')} className={`px-3 py-1 text-sm rounded-md transition-colors ${language === 'en' ? 'bg-white dark:bg-neutral-800 shadow-sm text-black dark:text-white' : 'text-neutral-500'}`}>EN</button>
                <button onClick={() => setLanguage('nl')} className={`px-3 py-1 text-sm rounded-md transition-colors ${language === 'nl' ? 'bg-white dark:bg-neutral-800 shadow-sm text-black dark:text-white' : 'text-neutral-500'}`}>NL</button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content - Expanded for very large screens */}
      <main className="max-w-[2400px] w-full px-4 md:px-8 2xl:px-16 mx-auto py-8 grid grid-cols-1 lg:grid-cols-12 gap-8 flex-1">
        
        {/* Left Column - Target */}
        <div className="lg:col-span-3 flex flex-col gap-8">
          <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl p-6 shadow-sm flex flex-col items-center">
            <h2 className="text-xl font-semibold mb-6">{currentT.targetLetter}</h2>
            {practiceMode === 'letter' ? (
              <div className="text-[12rem] 2xl:text-[14rem] font-bold leading-none tracking-tighter mb-6">{target.letter || '-'}</div>
            ) : (
              <div className="flex flex-col w-full gap-4 mb-6">
                {practiceMode === 'sentence' && (
                  <div className="flex gap-2">
                    <input 
                      type="text" 
                      placeholder="Type target sentence..."
                      value={targetSentenceInput}
                      onChange={e => setTargetSentenceInput(e.target.value)}
                      className="flex-1 bg-neutral-100 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-lg px-3 py-2 text-base outline-none focus:ring-2 focus:ring-neutral-500"
                    />
                    <button 
                      onClick={applyTargetSentence}
                      className="bg-black dark:bg-white text-white dark:text-black px-4 py-2 rounded-lg text-sm font-semibold hover:opacity-80 transition-opacity"
                    >Set</button>
                  </div>
                )}
                <div className="text-center font-medium text-neutral-500 mt-2 mb-2 text-2xl tracking-widest uppercase break-words">
                  {activeTargetSentence}
                </div>
                {practiceMode === 'sentence' && target.letter && target.letter !== '-' && (
                  <div className="flex items-center justify-center gap-4">
                    <span className="text-sm font-medium text-neutral-400 uppercase tracking-widest">Current:</span>
                    <span className="text-5xl font-bold">{target.letter}</span>
                  </div>
                )}
              </div>
            )}

            {target.image && (
              <div className="relative w-full aspect-[4/3] rounded-xl overflow-hidden border border-neutral-200 dark:border-neutral-800 mb-6 bg-neutral-100 dark:bg-black">
                <Image src={target.image} alt={`How to sign ${target.letter}`} fill unoptimized className="object-contain" />
              </div>
            )}
            
            {practiceMode === 'letter' && (
              <div className="flex gap-2 mt-auto">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className={`w-4 h-4 rounded-full border-2 transition-colors ${
                    dots.current > i * 3 
                      ? (dots.isMatch ? 'bg-emerald-500 border-emerald-500' : 'bg-black dark:bg-white border-black dark:border-white')
                      : 'bg-transparent border-neutral-300 dark:border-neutral-700'
                  }`} />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Center Column - Camera */}
        <div className="lg:col-span-6 flex flex-col gap-8">
          <WebcamView 
            isCameraActive={isCameraActive}
            onStartCamera={handleStartCamera}
            onFrameCapture={handleFrameCapture}
            status={status}
            isRecording={isRecording}
            progress={progress}
            startCameraText={currentT.startCamera}
          />

          <div className="flex gap-4">
            <button 
              disabled={!isCameraActive}
              onClick={toggleRecording}
              className={`flex-1 flex justify-center items-center gap-2 py-5 rounded-2xl font-bold text-lg transition-all ${
                isRecording 
                  ? 'bg-red-500 text-white hover:bg-red-600 animate-pulse' 
                  : 'bg-black dark:bg-white text-white dark:text-black hover:opacity-80 disabled:opacity-50 disabled:hover:opacity-50'
              }`}
            >
              {isRecording ? <Square className="w-6 h-6 fill-current" /> : <Play className="w-6 h-6 fill-current" />}
              {isRecording ? currentT.stopRecording : currentT.record}
            </button>
            
            <button 
              disabled={!isCameraActive || practiceMode !== 'letter'}
              onClick={skipLetter}
              className="px-8 py-5 rounded-2xl font-semibold text-lg bg-neutral-100 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 hover:bg-neutral-200 dark:hover:bg-neutral-800 disabled:opacity-50 transition-colors flex items-center gap-2 text-neutral-600 dark:text-neutral-400"
            >
              <SkipForward className="w-6 h-6" />
              {currentT.skipLetter}
            </button>
          </div>
          
          {(practiceMode === 'sentence' || practiceMode === 'free') && (
            <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl p-8 shadow-sm flex flex-col items-center">
              <div className="flex justify-between w-full mb-6">
                <span className="text-sm font-semibold text-neutral-400 uppercase tracking-widest">Recognized</span>
                <button onClick={clearSentence} className="text-sm text-neutral-500 hover:text-black dark:hover:text-white transition-colors">Clear</button>
              </div>
              <div className="text-5xl 2xl:text-6xl font-bold tracking-tight min-h-[80px] flex flex-wrap justify-center gap-1">
                {recognizedSentence.length > 0 ? recognizedSentence.map((r, i) => (
                  <span key={i} className={r.correct ? 'text-emerald-500' : 'text-neutral-900 dark:text-neutral-100'}>{r.char}</span>
                )) : <span className="text-neutral-300 dark:text-neutral-700 italic font-medium">Start signing...</span>}
                <span className="w-6 border-b-4 border-black dark:border-white animate-pulse inline-block ml-1" />
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Stats */}
        <div className="lg:col-span-3 flex flex-col gap-8">
          <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl p-6 shadow-sm flex flex-col items-center justify-center min-h-[300px]">
            <h2 className="text-xl font-semibold mb-6">{currentT.prediction}</h2>
            <div className="text-[10rem] 2xl:text-[12rem] font-bold leading-none text-emerald-500">{prediction.letter}</div>
          </div>
          
          <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl p-8 shadow-sm flex flex-col gap-6 text-lg">
            <div className="flex justify-between items-center py-3 border-b border-neutral-100 dark:border-neutral-800">
              <span className="text-neutral-500 font-medium">{currentT.correct}</span>
              <span className="font-bold text-2xl">{stats.correct}</span>
            </div>
            <div className="flex justify-between items-center py-3 border-b border-neutral-100 dark:border-neutral-800">
              <span className="text-neutral-500 font-medium">{currentT.attempts}</span>
              <span className="font-bold text-2xl">{stats.attempts}</span>
            </div>
            <div className="flex justify-between items-center py-3 border-b border-neutral-100 dark:border-neutral-800">
              <span className="text-neutral-500 font-medium">{currentT.accuracy}</span>
              <span className="font-bold text-2xl">{stats.accuracy}%</span>
            </div>
            <div className="flex justify-between items-center py-3">
              <span className="text-neutral-500 font-medium">{currentT.timeRemaining}</span>
              <span className="font-bold text-2xl text-emerald-500">{stats.timeRemaining}</span>
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}
