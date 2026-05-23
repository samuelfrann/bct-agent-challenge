import { useState } from 'react'

function App() {
  const [activeTab, setActiveTab] = useState('taskA'); // 'taskA' or 'taskB'
  
  // --- TASK A STATE ---
  const [userContextA, setUserContextA] = useState('');
  const [businessContextA, setBusinessContextA] = useState('');
  const [loadingA, setLoadingA] = useState(false);
  const [resultA, setResultA] = useState(null);

  // --- TASK B STATE ---
  const [userContextB, setUserContextB] = useState('');
  const [loadingB, setLoadingB] = useState(false);
  const [resultB, setResultB] = useState(null);

  // --- HANDLERS ---
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const handleSimulateTaskA = async (e) => {
    e.preventDefault();
    setLoadingA(true);
    try {
      const response = await fetch(`${API_URL}/simulate-review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_persona: { description: userContextA },
          item: { name: businessContextA, categories: "Restaurant", city: "Lagos" }
        }),
      });
      const data = await response.json();
      setResultA(data); 
    } catch (error) {
      console.error("Gateway error:", error);
      alert("Error: Gateway server is offline.");
    } finally {
      setLoadingA(false);
    }
  }

  const handleSimulateTaskB = async (e) => {
    e.preventDefault();
    setLoadingB(true);
    try {
      const response = await fetch(`${API_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_persona: { description: userContextB },
          context: { intent: userContextB },
          k: 5
        }),
      });
      const data = await response.json();
      setResultB(data); 
    } catch (error) {
      console.error("API error:", error);
      alert("Error: " + error.message);
    } finally {
      setLoadingB(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 font-sans">
      
      {/* Top Navigation Bar */}
      <nav className="bg-blue-900 text-white shadow-md">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex space-x-8">
            <div className="py-4 font-bold text-xl tracking-wide border-r border-blue-800 pr-8">
              BCT x DSN Agent
            </div>
            <button 
              onClick={() => setActiveTab('taskA')}
              className={`py-4 px-2 font-semibold transition-colors border-b-4 ${activeTab === 'taskA' ? 'border-blue-400 text-blue-100' : 'border-transparent text-blue-300 hover:text-white'}`}
            >
              Task A: User Modeling
            </button>
            <button 
              onClick={() => setActiveTab('taskB')}
              className={`py-4 px-2 font-semibold transition-colors border-b-4 ${activeTab === 'taskB' ? 'border-blue-400 text-blue-100' : 'border-transparent text-blue-300 hover:text-white'}`}
            >
              Task B: Recommendation
            </button>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto p-8">
        
        {/* ========================================== */}
        {/* TASK A VIEW                                */}
        {/* ========================================== */}
        {activeTab === 'taskA' && (
          <div className="animate-fade-in">
            <header className="mb-8">
              <h1 className="text-3xl font-bold text-gray-900">Task A: User Modeling Agent</h1>
              <p className="text-gray-600 mt-2">Simulating localized Nigerian user reviews and ratings.</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <h2 className="text-xl font-semibold mb-4 text-gray-800">Agent Inputs</h2>
                <form onSubmit={handleSimulateTaskA} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">User Persona</label>
                    <textarea 
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                      rows="4" value={userContextA} onChange={(e) => setUserContextA(e.target.value)} required
                    ></textarea>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Business Context</label>
                    <textarea 
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                      rows="4" value={businessContextA} onChange={(e) => setBusinessContextA(e.target.value)} required
                    ></textarea>
                  </div>
                  <button type="submit" disabled={loadingA} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:bg-blue-400">
                    {loadingA ? 'Simulating...' : 'Run Agent Simulation'}
                  </button>
                </form>
              </div>

              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 flex flex-col">
                <h2 className="text-xl font-semibold mb-4 text-gray-800">Agent Output</h2>
                {resultA ? (
                  <div className="flex-1 flex flex-col space-y-6">
                    <div className="bg-blue-50 p-6 rounded-lg border border-blue-100 text-center">
                      <p className="text-sm text-blue-600 uppercase tracking-wide font-bold mb-1">Predicted Rating</p>
                      <p className="text-5xl font-extrabold text-blue-900">{resultA.rating} / 5.0</p>
                    </div>
                    <div className="flex-1 bg-gray-50 p-6 rounded-lg border border-gray-200">
                      <p className="text-sm text-gray-500 uppercase tracking-wide font-bold mb-2">Simulated Review</p>
                      <p className="text-gray-800 text-lg italic">"{resultA.review_text || resultA.review}"</p>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-gray-400 border-2 border-dashed border-gray-200 rounded-lg p-6 text-center">
                    Enter parameters to see the generated rating.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ========================================== */}
        {/* TASK B VIEW                                */}
        {/* ========================================== */}
        {activeTab === 'taskB' && (
          <div className="animate-fade-in">
            <header className="mb-8">
              <h1 className="text-3xl font-bold text-gray-900">Task B: Contextual Recommendation</h1>
              <p className="text-gray-600 mt-2">Agentic workflow reasoning for cold-start and cross-domain retrieval.</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              <div className="lg:col-span-4 bg-white p-6 rounded-xl shadow-sm border border-gray-200 h-fit">
                <h2 className="text-xl font-semibold mb-4 text-gray-800">Persona & Context</h2>
                <form onSubmit={handleSimulateTaskB} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">User Intent & History</label>
                    <textarea 
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                      rows="6" 
                      placeholder="e.g., Looking for a nice spot in VI for a first date. Loves seafood, budget is flexible..."
                      value={userContextB} onChange={(e) => setUserContextB(e.target.value)} required
                    ></textarea>
                  </div>
                  <button type="submit" disabled={loadingB} className="w-full bg-blue-900 hover:bg-blue-800 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:bg-blue-400">
                    {loadingB ? 'Reasoning & Ranking...' : 'Generate Recommendations'}
                  </button>
                </form>
              </div>

              <div className="lg:col-span-8 bg-white p-6 rounded-xl shadow-sm border border-gray-200">
                <h2 className="text-xl font-semibold mb-4 text-gray-800">Ranked Output (NDCG Optimized)</h2>
                {resultB && resultB.recommendations ? (
                  <div className="space-y-4">
                    {resultB.recommendations.map((item, i) => (
                      <div key={item.item} className="p-5 border border-gray-100 rounded-lg bg-gray-50 flex gap-4">
                        <div className="shrink-0 w-12 h-12 bg-blue-100 text-blue-800 font-bold text-xl rounded-full flex items-center justify-center">
                          #{i + 1}
                        </div>
                        <div>
                          <h3 className="text-lg font-bold text-gray-900">{item.name}</h3>
                          <span className="inline-block px-2 py-1 bg-gray-200 text-gray-700 text-xs font-semibold rounded mt-1 mb-2">
                            {item.categories}
                          </span>
                          <p className="text-sm text-gray-600"><strong className="text-gray-800">Agent Reasoning:</strong> {item.reason}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center text-gray-400 border-2 border-dashed border-gray-200 rounded-lg p-6 text-center">
                    Enter the user's intent to view ranked recommendations and agentic reasoning.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

export default App