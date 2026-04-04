import React, { Suspense } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { ErrorBoundary } from '@/components/ui/ErrorBoundary'

// Lazy-loaded pages for code splitting
const OverviewPage = React.lazy(() => import('@/pages/OverviewPage').then(m => ({ default: m.OverviewPage })))
const ReconciliationPage = React.lazy(() => import('@/pages/ReconciliationPage').then(m => ({ default: m.ReconciliationPage })))
const WalkForwardPage = React.lazy(() => import('@/pages/WalkForwardPage').then(m => ({ default: m.WalkForwardPage })))
const SignalAnalyticsPage = React.lazy(() => import('@/pages/SignalAnalyticsPage').then(m => ({ default: m.SignalAnalyticsPage })))
const RegimeLabPage = React.lazy(() => import('@/pages/RegimeLabPage').then(m => ({ default: m.RegimeLabPage })))
const PortfolioLabPage = React.lazy(() => import('@/pages/PortfolioLabPage').then(m => ({ default: m.PortfolioLabPage })))
const MCSimPage = React.lazy(() => import('@/pages/MCSimPage').then(m => ({ default: m.MCSimPage })))
const StressTestPage = React.lazy(() => import('@/pages/StressTestPage').then(m => ({ default: m.StressTestPage })))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 2,
      retryDelay: 1_000,
    },
  },
})

function PageLoader() {
  return (
    <div className="flex items-center justify-center h-full min-h-[400px]">
      <LoadingSpinner size="lg" label="Loading page..." />
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route
                index
                element={
                  <Suspense fallback={<PageLoader />}>
                    <OverviewPage />
                  </Suspense>
                }
              />
              <Route
                path="reconciliation"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <ReconciliationPage />
                  </Suspense>
                }
              />
              <Route
                path="walk-forward"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <WalkForwardPage />
                  </Suspense>
                }
              />
              <Route
                path="signals"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <SignalAnalyticsPage />
                  </Suspense>
                }
              />
              <Route
                path="regimes"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <RegimeLabPage />
                  </Suspense>
                }
              />
              <Route
                path="portfolio"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <PortfolioLabPage />
                  </Suspense>
                }
              />
              <Route
                path="mc-sim"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <MCSimPage />
                  </Suspense>
                }
              />
              <Route
                path="stress-test"
                element={
                  <Suspense fallback={<PageLoader />}>
                    <StressTestPage />
                  </Suspense>
                }
              />
            </Route>
          </Routes>
        </ErrorBoundary>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
