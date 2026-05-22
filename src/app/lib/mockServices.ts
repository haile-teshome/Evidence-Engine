// This file used to host the mocked services for the frontend.
// All implementations now live in `./apiClient.ts` and call the real
// FastAPI backend at /api. We re-export here so existing imports from
// pages and components keep working without changes.

export {
  apiConfig,
  AIService,
  QualityService,
  DataAggregator,
  MetaAnalysisService,
  Deduplicator,
  ALL_SOURCES,
  AGENT_NAMES,
  formatDuration,
} from "./apiClient";

export type {
  Pico,
  Paper,
  Analysis,
  AgentVote,
  AgentTrace,
  ScreenResult,
  CriterionEvidence,
  FullTextResult,
  QualityIssue,
  QualityReport,
  StudyEffect,
  EffectMeasure,
  Tau2Method,
  PooledEstimate,
  MetaPoolResult,
  SubgroupResult,
  LOORow,
  CumulativeRow,
  FunnelData,
  EggerResult,
  BeggResult,
  TrimFillResult,
  MetaRegressionResult,
  MetaRunResult,
  ClarifyingQuestion,
} from "./apiClient";
