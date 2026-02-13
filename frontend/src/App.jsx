import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, User, Bot, Loader2, Settings, Download } from 'lucide-react';

function App() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  
  // Settings State
  const [config, setConfig] = useState({ avatars: [], voices: [] });
  const [selectedAvatar, setSelectedAvatar] = useState('');
  const [selectedVoice, setSelectedVoice] = useState('en-US-JennyNeural');
  const [showSettings, setShowSettings] = useState(false);

  const videoRef = useRef(null);

  // Fetch Available Avatars & Voices on Load
  useEffect(() => {
    async function fetchConfig() {
      try {
        const res = await axios.get('http://127.0.0.1:8000/config');
        setConfig(res.data);
        if (res.data.avatars.length > 0) setSelectedAvatar(res.data.avatars[0]);
      } catch (err) {
        console.error("Failed to load config", err);
      }
    }
    fetchConfig();
  }, []);

  const handleChat = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setVideoUrl(null);
    
    try {
      // Send avatar and voice choice to backend
      const url = `http://127.0.0.1:8000/chat?user_query=${encodeURIComponent(input)}&avatar_id=${selectedAvatar}&voice_id=${selectedVoice}`;
      const response = await axios.post(url);
      
      setChatHistory([...chatHistory, { type: 'user', text: input }, { type: 'bot', text: response.data.text }]);
      setVideoUrl(`http://127.0.0.1:8000/get-video?t=${new Date().getTime()}`);
      setInput('');
    } catch (error) {
      alert("Error: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-slate-900 flex items-center justify-center p-6 font-sans">
      <div className="w-full max-w-5xl h-[85vh] bg-slate-800 rounded-3xl overflow-hidden shadow-2xl flex flex-col border border-slate-700">
        
        {/* Header */}
        <header className="flex items-center justify-between p-5 border-b border-slate-700 bg-slate-800/80 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
              <Bot size={24} className="text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Sprout AI Tutor</h1>
              <p className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold">Customizable Engine</p>
            </div>
          </div>
          
          {/* Settings Toggle */}
          <button 
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-xl transition-all ${showSettings ? 'bg-green-500 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}
          >
            <Settings size={20} />
          </button>
        </header>

        {/* Settings Panel (Collapsible) */}
        {showSettings && (
          <div className="bg-slate-900/50 p-4 border-b border-slate-700 flex gap-4 items-center animate-in slide-in-from-top-2">
            <div className="flex-1">
              <label className="text-xs text-slate-400 font-bold uppercase block mb-1">Avatar</label>
              <select 
                value={selectedAvatar} 
                onChange={(e) => setSelectedAvatar(e.target.value)}
                className="w-full bg-slate-800 text-white text-sm rounded-lg p-2 border border-slate-600 focus:border-green-500 outline-none"
              >
                {config.avatars.map(av => <option key={av} value={av}>{av}</option>)}
              </select>
            </div>
            <div className="flex-1">
              <label className="text-xs text-slate-400 font-bold uppercase block mb-1">Voice</label>
              <select 
                value={selectedVoice} 
                onChange={(e) => setSelectedVoice(e.target.value)}
                className="w-full bg-slate-800 text-white text-sm rounded-lg p-2 border border-slate-600 focus:border-green-500 outline-none"
              >
                {config.voices.map(v => <option key={v.id} value={v.id}>{v.name}</option>)}
              </select>
            </div>
          </div>
        )}

        {/* Body */}
        <div className="flex-1 flex overflow-hidden p-6 gap-6">
          <section className="flex-[1.4] bg-black rounded-2xl overflow-hidden border border-slate-700 relative shadow-inner group">
            {videoUrl ? (
              <video ref={videoRef} src={videoUrl} autoPlay className="w-full h-full object-contain" />
            ) : (
              <div className="h-full w-full flex flex-col items-center justify-center text-slate-600">
                <User size={80} className="opacity-10 mb-2" />
                <p className="text-xs font-medium uppercase opacity-40">
                  {loading ? "Synthesizing..." : `Ready: ${selectedAvatar}`}
                </p>
              </div>
            )}
             {loading && (
              <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                  <Loader2 className="animate-spin text-green-500" size={48} />
                  <span className="text-green-500 font-bold text-xs tracking-widest animate-pulse">GENERATING VIDEO</span>
                </div>
              </div>
            )}
          </section>

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
            <div className="mt-4 relative">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleChat()}
                placeholder="Ask Sprout anything..."
                className="w-full bg-slate-900 border border-slate-700 rounded-2xl py-4 px-5 pr-14 text-white focus:outline-none focus:border-green-500"
              />
              <button onClick={handleChat} disabled={loading} className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 bg-green-500 hover:bg-green-400 text-white rounded-xl">
                <Send size={18} />
              </button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export default App;