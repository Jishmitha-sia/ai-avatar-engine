import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, User, Bot, Loader2, Settings, Download, RotateCcw, Mic, MicOff, UploadCloud, DownloadCloud, Star, Maximize2, Minimize2 } from 'lucide-react';

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
  const [concepts, setConcepts] = useState([]); // <--- NEW STATE FOR BOARD
  const [quizData, setQuizData] = useState([]);
  const [showQuiz, setShowQuiz] = useState(false);
  const [currentQuizIndex, setCurrentQuizIndex] = useState(0);
  const [quizScore, setQuizScore] = useState(0);
  const [quizFinished, setQuizFinished] = useState(false);
  const [isWide, setIsWide] = useState(false); // <--- NEW TOGGLE

  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  // Fetch Config
  useEffect(() => {
    async function fetchConfig() {
      try {
        const res = await axios.get('http://127.0.0.1:8000/config');
        setConfig(res.data);
        if (res.data.avatars.length > 0) {
          const defaultAv = res.data.avatars.find(a => a === "womantutor.jpg") || res.data.avatars[0];
          setSelectedAvatar(defaultAv);
        }
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
      if (response.data.concepts) setConcepts(response.data.concepts);
      setVideoUrl(`http://127.0.0.1:8000/get-video?t=${new Date().getTime()}`);
      setInput('');
    } catch (error) {
      alert("Error: " + error.message);
      setVideoUrl(null); // Revert to idle on error
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.type !== "application/pdf") { alert("Only PDF files are supported!"); return; }
    
    setLoading(true);
    setVideoUrl('loading');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const url = `http://127.0.0.1:8000/pdf-to-video?avatar_id=${selectedAvatar}&voice_id=${selectedVoice}`;
      const response = await axios.post(url, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setChatHistory([...chatHistory, { type: 'user', text: `Uploaded PDF: ${file.name}` }, { type: 'bot', text: response.data.text }]);
      if (response.data.concepts) setConcepts(response.data.concepts);
      setVideoUrl(`http://127.0.0.1:8000/get-video?t=${new Date().getTime()}`);
    } catch (error) {
      alert("Error: " + (error.response?.data?.detail || error.message));
      setVideoUrl(null);
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleVideoEnd = () => {
    setVideoUrl(null); 
  };

  const startQuiz = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://127.0.0.1:8000/generate-quiz');
      if (res.data.length > 0) {
        setQuizData(res.data);
        setCurrentQuizIndex(0);
        setQuizScore(0);
        setShowQuiz(true);
      } else {
        alert("Not enough context yet. Chat more or upload a PDF!");
      }
    } catch (err) { alert("Quiz Error: " + err.message); }
    finally { setLoading(false); }
  };

  const handleQuizAnswer = (selected) => {
    const isCorrect = selected === quizData[currentQuizIndex].answer;
    if (isCorrect) setQuizScore(quizScore + 1);
    
    if (currentQuizIndex + 1 < quizData.length) {
      setCurrentQuizIndex(currentQuizIndex + 1);
    } else {
      setQuizFinished(true);
    }
  };

  const downloadStudyGuide = () => {
    if (concepts.length === 0) return;
    const content = `SPROUT STUDY GUIDE\n==================\n\n` + 
      concepts.map(c => `[${c.title}]\n${c.explanation}\n`).join('\n') +
      `\nGenerated on: ${new Date().toLocaleString()}`;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "Sprout_Study_Guide.txt";
    a.click();
  };

  return (
    <div className="min-h-screen w-full bg-slate-950 flex items-center justify-center p-2 sm:p-4 font-sans selection:bg-green-500/30">
      <div 
        className="w-full h-[96vh] bg-slate-900 overflow-hidden shadow-2xl flex flex-col border border-slate-800 rounded-3xl relative transition-all duration-700 ease-in-out"
        style={{ maxWidth: isWide ? '98vw' : '1500px' }}
      >
        
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
          <div className="flex gap-2">
            <button onClick={() => setIsWide(!isWide)} title={isWide ? "Focus Mode" : "Widescreen Mode"} className="p-2 bg-slate-700 text-slate-300 rounded-xl hover:bg-slate-600 transition-colors">
              {isWide ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
            </button>
            <input type="file" accept=".pdf" ref={fileInputRef} onChange={handleFileUpload} className="hidden" />
            <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-2 p-2 px-3 bg-green-600 hover:bg-green-500 text-white rounded-xl font-medium shadow-md transition-colors" disabled={loading}>
              <UploadCloud size={20} />
            </button>
            <button onClick={() => setShowSettings(!showSettings)} className="p-2 bg-slate-700 text-slate-300 rounded-xl hover:bg-slate-600">
              <Settings size={20} />
            </button>
          </div>
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
          <section className="flex-[1.2] bg-black rounded-2xl overflow-hidden border border-slate-700 relative shadow-inner group flex items-center justify-center">
            
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

          {/* KNOWLEDGE BOARD PANEL */}
          <section className="flex-[1.2] bg-slate-900/40 rounded-2xl border border-slate-700/50 flex flex-col overflow-hidden">
             <div className="p-4 border-b border-slate-700/50 bg-slate-800/30 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                  <h3 className="text-[11px] font-bold uppercase tracking-widest text-slate-400">Knowledge Deep-Dive (10 Points)</h3>
                </div>
                {concepts.length > 0 && (
                  <div className="flex gap-2">
                    <button onClick={downloadStudyGuide} title="Export Guide" className="text-slate-400 hover:text-white transition-colors">
                      <DownloadCloud size={14} />
                    </button>
                    <button onClick={startQuiz} className="text-[9px] font-bold bg-green-600/20 hover:bg-green-600/40 text-green-400 px-2 py-1 rounded-lg border border-green-500/20 transition-all uppercase tracking-tighter">
                      Start Quiz
                    </button>
                  </div>
                )}
             </div>
             <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                {concepts.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center opacity-30 mt-10">
                    <RotateCcw className="mb-2" size={32} />
                    <p className="text-[10px] uppercase font-bold tracking-tighter">Waiting for insights...</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-3 pb-4">
                    {concepts.map((c, i) => (
                      <div key={i} className="bg-slate-800/80 border border-slate-600/50 p-3 rounded-xl shadow-lg animate-in fade-in slide-in-from-right-4 duration-500 hover:border-green-500/50 transition-colors group" style={{ animationDelay: `${i * 100}ms` }}>
                        <h4 className="text-green-400 font-bold text-[10px] mb-1 uppercase tracking-wider group-hover:text-green-300 transition-colors">{c.title}</h4>
                        <p className="text-white text-[10px] leading-relaxed opacity-80">{c.explanation}</p>
                      </div>
                    ))}
                  </div>
                )}
             </div>
          </section>
        </div>

        {/* QUIZ OVERLAY */}
        {showQuiz && (
          <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-md flex items-center justify-center p-6 text-center">
            <div className="bg-slate-900 border border-slate-700 p-8 rounded-3xl max-w-md w-full shadow-2xl animate-in fade-in zoom-in-95 duration-500">
              
              {!quizFinished ? (
                <>
                  <div className="flex justify-between items-center mb-6 text-left">
                    <span className="text-green-500 font-bold text-xs uppercase tracking-widest">Question {currentQuizIndex + 1}/{quizData.length}</span>
                    <button onClick={() => setShowQuiz(false)} className="text-slate-500 hover:text-white">✕</button>
                  </div>
                  <h2 className="text-xl font-bold text-white mb-8 leading-tight text-left">{quizData[currentQuizIndex].question}</h2>
                  <div className="space-y-3">
                    {quizData[currentQuizIndex].options.map((opt, i) => (
                      <button 
                        key={i} 
                        onClick={() => handleQuizAnswer(opt)}
                        className="w-full text-left p-4 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl text-slate-200 text-sm transition-all active:scale-95"
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <div className="py-6">
                  <div className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-6 text-green-500">
                    <Star size={40} />
                  </div>
                  <h2 className="text-2xl font-bold text-white mb-2">Knowledge Master!</h2>
                  <p className="text-slate-400 mb-8">You scored {quizScore} out of {quizData.length}</p>
                  <button 
                    onClick={() => { setQuizFinished(false); setShowQuiz(false); }}
                    className="w-full p-4 bg-green-600 hover:bg-green-500 text-white rounded-xl font-bold transition-all"
                  >
                    Back to Dashboard
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* CSS Animation for Breathing */}
        <style>{`
          @keyframes breathe {
            0%, 100% { transform: scale(1); opacity: 0.9; }
            50% { transform: scale(1.02); opacity: 1; }
          }
          @keyframes slideInRight {
            from { transform: translateX(30px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
          }
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          .animate-breathe {
            animation: breathe 4s ease-in-out infinite;
          }
          .animate-in {
            animation: fadeIn 0.5s ease-out forwards, slideInRight 0.5s ease-out forwards;
          }
        `}</style>
      </div>
    </div>
  );
}

export default App;