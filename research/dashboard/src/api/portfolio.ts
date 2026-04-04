import { api } from './client'
import type { PortfolioWeight, EfficientFrontierPoint, RiskContribution } from '@/types/portfolio'
import {
  generateMockPortfolioWeights,
  generateMockEfficientFrontier,
  generateMockRiskContribution,
} from './mockData'

const USE_MOCK = true

export async function fetchPortfolioWeights(): Promise<PortfolioWeight[]> {
  if (USE_MOCK) return generateMockPortfolioWeights()
  return api.get<PortfolioWeight[]>('/api/portfolio/weights')
}

export async function fetchEfficientFrontier(): Promise<EfficientFrontierPoint[]> {
  if (USE_MOCK) return generateMockEfficientFrontier()
  return api.get<EfficientFrontierPoint[]>('/api/portfolio/frontier')
}

export async function fetchRiskContribution(): Promise<RiskContribution[]> {
  if (USE_MOCK) return generateMockRiskContribution()
  return api.get<RiskContribution[]>('/api/portfolio/risk-contribution')
}
