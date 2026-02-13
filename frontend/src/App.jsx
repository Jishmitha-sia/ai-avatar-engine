import React, { useState, useRef } from 'react';
import axios from 'axios';
import { Send, User, Bot, Loader2 } from 'lucide-react';

function App() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const videoRef = useRef(null);

  const handleChat = async () => {
    if (!input.trim()) return;
    
    setLoading(true);
    setVideoUrl(null);
    
    try {
      const response = await axios.post(`http://127.0.0.1:8000/chat?user_query=${encodeURIComponent(input)}`);
      setChatHistory([...chatHistory, { type: 'user', text: input }, { type: 'bot', text: response.data.text }]);
      setVideoUrl(`http://127.0.0.1:8000/get-video?t=${new Date().getTime()}`);
      setInput('');
    } catch (error) {
      console.error("Error communicating with Sprout:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    // Outer Wrapper: Centers the entire application in the middle of the screen
    <div className="min-h-screen w-full bg-slate-900 flex items-center justify-center p-6">
      
      {/* Centered Dashboard Container */}
      <div className="w-full max-w-5xl h-[80vh] bg-slate-800 rounded-3xl overflow-hidden shadow-2xl flex flex-col border border-slate-700">
        
        {/* Header */}
        <header className="flex items-center gap-3 p-5 border-b border-slate-700 bg-slate-800/80 backdrop-blur-md">
          <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg shadow-green-500/20">
            <Bot size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white leading-tight">Sprout AI Tutor</h1>
            <p className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold">Conversational Prototype</p>
          </div>
        </header>

        {/* Main Content Area */}
        <div className="flex-1 flex overflow-hidden p-6 gap-6">
          
          {/* Left Side: Avatar Viewport */}
          <section className="flex-[1.4] bg-black rounded-2xl overflow-hidden border border-slate-700 relative shadow-inner">
            {videoUrl ? (
              <video 
                ref={videoRef}
                src={videoUrl} 
                autoPlay 
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="h-full w-full flex flex-col items-center justify-center text-slate-600">
                <User size={80} className="opacity-10 mb-2" />
                <p className="text-xs font-medium uppercase tracking-tighter opacity-40">
                  {loading ? "Generating Visuals..." : "Awaiting Input"}
                </p>
              </div>
            )}
            
            {loading && (
              <div className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center">
                <div className="flex flex-col items-center gap-4">
                  <Loader2 className="animate-spin text-green-500" size={48} />
                  <span className="text-green-500 font-bold text-xs tracking-widest animate-pulse">PROCESSING FRAMES</span>
                </div>
              </div>
            )}
          </section>

          {/* Right Side: Chat Column */}
          <section className="flex-1 flex flex-col h-full overflow-hidden">
            {/* Scrollable Chat History */}
            <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar">
              {chatHistory.length === 0 && (
                <div className="h-full flex items-center justify-center text-center px-4">
                  <p className="text-slate-500 text-xs italic">Ask Sprout a question to begin the lesson.</p>
                </div>
              )}
              {chatHistory.map((msg, i) => (
                <div key={i} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] p-3 rounded-2xl text-sm leading-relaxed shadow-sm ${
                    msg.type === 'user' 
                    ? 'bg-blue-600 text-white rounded-tr-none' 
                    : 'bg-slate-700 text-slate-100 rounded-tl-none border border-slate-600'
                  }`}>
                    {msg.text}
                  </div>
                </div>
              ))}
            </div>

            {/* Message Input Box */}
            <div className="mt-4 relative">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleChat()}
                placeholder="Type your message..."
                className="w-full bg-slate-900 border border-slate-700 rounded-2xl py-4 px-5 pr-14 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-green-500/30 focus:border-green-500 transition-all"
              />
              <button 
                onClick={handleChat}
                disabled={loading}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 bg-green-500 hover:bg-green-400 text-white rounded-xl shadow-lg transition-all disabled:opacity-30 disabled:grayscale"
              >
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