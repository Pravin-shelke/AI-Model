/**
 * Balaji Framework AI Service
 * Integrates XGBoost AI predictions into React Native app
 * 
 * Copy this file to: src/services/ML/BalajiAIService.ts
 */

import axios, { AxiosInstance } from 'axios';

// Configuration
const AI_API_URL = __DEV__ 
  ? 'http://localhost:5001/api/v1'  // Development (iOS Simulator) - Port 5001
  : 'http://10.0.2.2:5001/api/v1';  // Android Emulator - Port 5001
  // For production, use your actual server URL

interface AssessmentInput {
  country: string;
  crop: string;
  partner: string;
  irrigation: string;
  hired_workers: string;
  area: number;
}

interface PredictionResult {
  value: string;
  confidence: number;
}

interface PredictionResponse {
  success: boolean;
  predictions: Record<string, PredictionResult>;
  statistics: {
    total_indicators: number;
    high_confidence: number;
    medium_confidence: number;
    low_confidence: number;
    average_confidence: number;
  };
  metadata: AssessmentInput;
}

interface BatchPredictionRequest {
  assessments: AssessmentInput[];
}

interface BatchPredictionResponse {
  success: boolean;
  total: number;
  results: Array<{
    index: number;
    success: boolean;
    predictions?: Record<string, PredictionResult>;
    error?: string;
  }>;
}

class BalajiAIService {
  private apiClient: AxiosInstance;

  constructor() {
    this.apiClient = axios.create({
      baseURL: AI_API_URL,
      timeout: 30000, // 30 seconds
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Check if AI service is available
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await axios.get(`${AI_API_URL.replace('/api/v1', '')}/health`);
      return response.data.status === 'healthy';
    } catch (error) {
      console.error('AI Service health check failed:', error);
      return false;
    }
  }

  /**
   * Predict all assessment indicators based on 6 basic inputs
   * Reduces 15-20 minute questionnaire to 2-3 minutes
   */
  async predictAssessment(input: AssessmentInput): Promise<PredictionResponse> {
    try {
      console.log('🤖 Calling AI for predictions...', input);
      
      const response = await this.apiClient.post<PredictionResponse>('/predict', input);
      
      console.log(`✅ AI predictions received: ${response.data.statistics.total_indicators} indicators`);
      console.log(`   High confidence: ${response.data.statistics.high_confidence}`);
      console.log(`   Average confidence: ${response.data.statistics.average_confidence}%`);
      
      return response.data;
    } catch (error: any) {
      console.error('❌ AI prediction failed:', error.response?.data || error.message);
      throw new Error(error.response?.data?.error || 'Failed to get AI predictions');
    }
  }

  /**
   * Predict assessments for multiple farmers at once
   */
  async predictBatch(request: BatchPredictionRequest): Promise<BatchPredictionResponse> {
    try {
      console.log(`🤖 Batch prediction for ${request.assessments.length} assessments...`);
      
      const response = await this.apiClient.post<BatchPredictionResponse>('/predict/batch', request);
      
      const successCount = response.data.results.filter(r => r.success).length;
      console.log(`✅ Batch complete: ${successCount}/${response.data.total} successful`);
      
      return response.data;
    } catch (error: any) {
      console.error('❌ Batch prediction failed:', error.response?.data || error.message);
      throw new Error(error.response?.data?.error || 'Failed to get batch predictions');
    }
  }

  /**
   * Get list of all indicators that can be predicted
   */
  async getIndicators(): Promise<string[]> {
    try {
      const response = await this.apiClient.get<{ success: boolean; indicators: string[] }>('/indicators');
      return response.data.indicators;
    } catch (error: any) {
      console.error('❌ Failed to get indicators:', error.message);
      throw new Error('Failed to get indicators list');
    }
  }

  /**
   * Format predictions for assessment form
   * Converts AI predictions into format compatible with your existing assessment structure
   */
  formatPredictionsForForm(predictions: Record<string, PredictionResult>): Record<string, any> {
    const formData: Record<string, any> = {};
    
    for (const [indicator, result] of Object.entries(predictions)) {
      // Map indicator to form field
      formData[indicator] = {
        value: result.value,
        confidence: result.confidence,
        isAIPredicted: true,
        needsReview: result.confidence < 60, // Flag low-confidence predictions for review
      };
    }
    
    return formData;
  }
}

// Export singleton instance
export const balajiAIService = new BalajiAIService();

// Export types
export type {
  AssessmentInput,
  PredictionResult,
  PredictionResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
};

export default BalajiAIService;
