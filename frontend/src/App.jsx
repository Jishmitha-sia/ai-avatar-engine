import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, User, Bot, Loader2, Settings, Download, RotateCcw, Mic, MicOff } from 'lucide-react';

function App() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null); // If null, show IDLE state
  const [chatHistory, setChatHistory] = useState([]);
  
  // Voice & Settings State
  const [isListening, setIsListening] = useState(false);
  const [config, setConfig] = useState({ avatars: [], voices: [] });
  const [selectedAvatar, setSelectedAvatar] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('en-US-JennyNeural');
  const [showSettings, setShowSettings] = useState(false);

  const videoRef = useRef(null);

  // Fetch Config
  useEffect(() => {
    async function fetchConfig() {
      try {
        const res = await axios.get('http://127.0.0.1:8000/config');
        setConfig(res.data);
        if (res.data.avatars.length > 0) setSelectedAvatar(res.data.avatars[0]);
      } catch (err) { console.error("Config Error", err); }
    }
    fetchConfig();
  }, []);

  // Voice Input
  const handleVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { alert("Browser not supported"); return; }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onresult = (e) => setInput(e.results[0][0].transcript);
    recognition.start();
  };

  const handleChat = async (manualInput = null) => {
    const query = manualInput || input;
    if (!query.trim()) return;
    setLoading(true);
    
    // Switch to video mode immediately to show loading spinner
    setVideoUrl('loading'); 
    
    try {
      const url = `http://127.0.0.1:8000/chat?user_query=${encodeURIComponent(query)}&avatar_id=${selectedAvatar}&voice_id=${selectedVoice}`;
      const response = await axios.post(url);
      setChatHistory([...chatHistory, { type: 'user', text: query }, { type: 'bot', text: response.data.text }]);
      setVideoUrl(`http://127.0.0.1:8000/get-video?t=${new Date().getTime()}`);
      setInput('');
    } catch (error) {
      alert("Error: " + error.message);
      setVideoUrl(null); // Revert to idle on error
    } finally {
      setLoading(false);
    }
  };

  // Switch back to idle when video ends
  const handleVideoEnd = () => {
    setVideoUrl(null); 
  };

  return (
    <div className="min-h-screen w-full bg-slate-900 flex items-center justify-center p-6 font-sans">
      <div className="w-full max-w-5xl h-[85vh] bg-slate-800 rounded-3xl overflow-hidden shadow-2xl flex flex-col border border-slate-700">
        
        {/* Header */}
        <header className="flex items-center justify-between p-5 border-b border-slate-700 bg-slate-800/80 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center shadow-lg transition-colors ${isListening ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}>
              {isListening ? <Mic size={24} className="text-white" /> : <Bot size={24} className="text-white" />}
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Sprout AI Tutor</h1>
              <p className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold">
                {isListening ? "Listening..." : "Interactive Avatar"}
              </p>
            </div>
          </div>
          <button onClick={() => setShowSettings(!showSettings)} className="p-2 bg-slate-700 text-slate-300 rounded-xl hover:bg-slate-600">
            <Settings size={20} />
          </button>
        </header>

        {/* Settings */}
        {showSettings && (
          <div className="bg-slate-900/50 p-4 border-b border-slate-700 flex gap-4 animate-in slide-in-from-top-2">
            <select value={selectedAvatar} onChange={(e) => setSelectedAvatar(e.target.value)} className="flex-1 bg-slate-800 text-white text-sm rounded-lg p-2 border border-slate-600">
              {config.avatars.map(av => <option key={av} value={av}>{av}</option>)}
            </select>
            <select value={selectedVoice} onChange={(e) => setSelectedVoice(e.target.value)} className="flex-1 bg-slate-800 text-white text-sm rounded-lg p-2 border border-slate-600">
              {config.voices.map(v => <option key={v.id} value={v.id}>{v.name}</option>)}
            </select>
          </div>
        )}

        {/* Body */}
        <div className="flex-1 flex overflow-hidden p-6 gap-6">
          
          {/* AVATAR VIEWPORT */}
          <section className="flex-[1.4] bg-black rounded-2xl overflow-hidden border border-slate-700 relative shadow-inner group flex items-center justify-center">
            
            {/* 1. TALKING STATE (Video) */}
            {videoUrl && videoUrl !== 'loading' && (
              <video 
                src={videoUrl} 
                autoPlay 
                onEnded={handleVideoEnd}
                className="w-full h-full object-contain"
              />
            )}

            {/* 2. LOADING STATE */}
            {loading && (
              <div className="absolute inset-0 z-20 bg-black/60 backdrop-blur-sm flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                  <Loader2 className="animate-spin text-green-500" size={48} />
                  <span className="text-green-500 font-bold text-xs tracking-widest animate-pulse">GENERATING...</span>
                </div>
              </div>
            )}

            {/* 3. IDLE STATE (Breathing Image) */}
            {(!videoUrl || videoUrl === 'loading') && selectedAvatar && (
              <div className="w-full h-full flex items-center justify-center bg-gradient-to-b from-slate-800 to-black">
                <img 
                  src={`http://127.0.0.1:8000/avatars/${selectedAvatar}`} 
                  alt="Avatar"
                  className="max-h-full max-w-full object-contain animate-breathe filter brightness-90 hover:brightness-110 transition-all duration-500"
                  style={{ 
                    animation: "breathe 4s ease-in-out infinite"
                  }} 
                />
              </div>
            )}

            {/* Helper Text if no avatar */}
            {!selectedAvatar && !loading && (
              <div className="text-slate-600 flex flex-col items-center">
                <User size={60} className="opacity-20 mb-2"/>
                <p className="text-xs uppercase">No Avatar Selected</p>
              </div>
            )}
          </section>

          {/* CHAT SECTION */}
          <section className="flex-1 flex flex-col h-full overflow-hidden">
            <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar">
              {chatHistory.map((msg, i) => (
                <div key={i} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] p-3 rounded-2xl text-sm ${msg.type === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-100'}`}>
                    {msg.text}
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-4 relative flex items-center gap-2">
              <button onClick={handleVoiceInput} className={`p-4 rounded-2xl transition-all shadow-lg ${isListening ? 'bg-red-500 text-white animate-pulse' : 'bg-slate-700 text-slate-300'}`}>
                {isListening ? <MicOff size={20} /> : <Mic size={20} />}
              </button>
              <div className="relative flex-1">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleChat(null)}
                  placeholder={isListening ? "Listening..." : "Type a message..."}
                  className="w-full bg-slate-900 border border-slate-700 rounded-2xl py-4 px-5 pr-14 text-white focus:outline-none focus:border-green-500"
                />
                <button onClick={() => handleChat(null)} disabled={loading} className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 bg-green-500 hover:bg-green-400 text-white rounded-xl">
                  <Send size={18} />
                </button>
              </div>
            </div>
          </section>
        </div>

        {/* CSS Animation for Breathing */}
        <style>{`
          @keyframes breathe {
            0%, 100% { transform: scale(1); opacity: 0.9; }
            50% { transform: scale(1.02); opacity: 1; }
          }
          .animate-breathe {
            animation: breathe 4s ease-in-out infinite;
          }
        `}</style>
      </div>
    </div>
  );
}

export default App;