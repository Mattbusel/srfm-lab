import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { LiveProvider } from './context/LiveContext';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Backtest } from './pages/Backtest';
import { MonteCarlo } from './pages/MonteCarlo';
import { Sensitivity } from './pages/Sensitivity';
import { Correlation } from './pages/Correlation';
import { Archaeology } from './pages/Archaeology';
import { Replay } from './pages/Replay';
import { Reports } from './pages/Reports';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <LiveProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout><Dashboard /></Layout>} />
            <Route path="/backtest" element={<Layout><Backtest /></Layout>} />
            <Route path="/montecarlo" element={<Layout><MonteCarlo /></Layout>} />
            <Route path="/sensitivity" element={<Layout><Sensitivity /></Layout>} />
            <Route path="/correlation" element={<Layout><Correlation /></Layout>} />
            <Route path="/archaeology" element={<Layout><Archaeology /></Layout>} />
            <Route path="/replay" element={<Layout><Replay /></Layout>} />
            <Route path="/reports" element={<Layout><Reports /></Layout>} />
          </Routes>
        </BrowserRouter>
      </LiveProvider>
    </QueryClientProvider>
  );
}
