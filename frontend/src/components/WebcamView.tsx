"use client";

import React, { useEffect, useRef, memo } from 'react';
import { Camera } from 'lucide-react';

interface WebcamViewProps {
  isCameraActive: boolean;
  onStartCamera: () => void;
  onFrameCapture: (frameData: string) => void;
  status: { text: string; connected: boolean; handDetected: boolean };
  isRecording: boolean;
  progress: number;
  startCameraText: string;
}

export const WebcamView = memo(function WebcamView({
  isCameraActive,
  onStartCamera,
  onFrameCapture,
  status,
  isRecording,
  progress,
  startCameraText
}: WebcamViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Expose the video stream when activated
  useEffect(() => {
    if (isCameraActive && videoRef.current && !videoRef.current.srcObject) {
      navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480, facingMode: 'user' } 
      }).then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      }).catch(err => {
        console.error('Error accessing webcam', err);
      });
    }
  }, [isCameraActive]);

  // Frame capture loop runs constantly while camera is active
  useEffect(() => {
    let captureTimeout: NodeJS.Timeout;

    const captureFrame = () => {
      if (!isCameraActive || !videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      if (!video.videoWidth) return; // Video not ready

      const context = canvas.getContext('2d');
      if (!context) return;

      const MAX_WIDTH = 320;
      const scale = Math.min(1.0, MAX_WIDTH / video.videoWidth);
      canvas.width = video.videoWidth * scale;
      canvas.height = video.videoHeight * scale;
      
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frameData = canvas.toDataURL('image/jpeg', 0.6);

      if (frameData.length > 100 && frameData !== 'data:,') {
        onFrameCapture(frameData);
      }
    };

    const loop = () => {
      captureFrame();
      captureTimeout = setTimeout(loop, 100);
    };

    if (isCameraActive) {
      loop();
    }

    return () => clearTimeout(captureTimeout);
  }, [isCameraActive, onFrameCapture]);

  return (
    <div className="relative bg-black rounded-2xl overflow-hidden aspect-[4/3] border border-neutral-200 dark:border-neutral-800 shadow-sm flex items-center justify-center">
      {!isCameraActive && (
        <button onClick={onStartCamera} className="absolute z-10 bg-white text-black px-6 py-3 rounded-full font-semibold flex items-center gap-2 hover:scale-105 transition-transform">
          <Camera className="w-5 h-5" />
          {startCameraText}
        </button>
      )}
      <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover -scale-x-100" />
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Status Overlay */}
      <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md text-white border border-white/10 px-4 py-2 rounded-full text-sm font-medium flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${status.connected ? (status.handDetected ? 'bg-emerald-400 animate-pulse' : 'bg-white') : 'bg-red-500'}`} />
        {status.text}
      </div>
      
      {/* Progress Overlay */}
      {isRecording && progress > 0 && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-black/60 backdrop-blur-md text-white border border-white/20 px-8 py-4 rounded-full text-2xl font-bold shadow-2xl">
          {Math.round(progress)}%
        </div>
      )}
    </div>
  );
});
