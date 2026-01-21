/**
 * Assessment Framework AI Service - IMPROVED VERSION v2.0
 * Now supports dynamic question loading from API
 * 
 * Copy this file to: src/services/ML/AssessmentAIService.ts
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

// NEW: Question metadata interfaces
interface Question {
  indicatorCode: string;
  description: string;
  displayId: string;
  categoryId: string;
  categoryName: string;
  type: string;
  required: boolean;
  options: string[];
  helpText?: string;
  aiAvailable: boolean;
  aiTrainingSamples?: number;
}

interface QuestionsResponse {
  success: boolean;
  total_questions: number;
  total_categories: number;
  version: string;
  lastUpdated: string;
  questions: Question[];
  categories: Array<{
    categoryId: string;
    categoryName: string;
    displayId: string;
    type: string;
    questions: Question[];
  }>;
}

interface QuestionDetailsResponse {
  success: boolean;
  question: Question & {
    aiInfo: {
      trained: boolean;
      possibleValues?: string[];
      trainingSamples?: number;
      reason?: string;
    };
  };
}

class AssessmentAIService {
  private apiClient: AxiosInstance;
  private questionsCache: Map<string, Question> | null = null;
  private questionsCacheExpiry: number = 0;
  private readonly CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

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
   * Load all questions from API (with caching)
   * Use this instead of hardcoding questions in the app
   */
  async loadQuestions(forceRefresh = false): Promise<Question[]> {
    // Return cache if still valid
    if (!forceRefresh && this.questionsCache && Date.now() < this.questionsCacheExpiry) {
      console.log('📋 Returning cached questions');
      return Array.from(this.questionsCache.values());
    }

    try {
      console.log('📥 Loading questions from API...');
      const response = await this.apiClient.get<QuestionsResponse>('/questions');
      
      // Update cache
      this.questionsCache = new Map(
        response.data.questions.map(q => [q.indicatorCode, q])
      );
      this.questionsCacheExpiry = Date.now() + this.CACHE_DURATION;
      
      console.log(`✅ Loaded ${response.data.total_questions} questions from ${response.data.total_categories} categories`);
      console.log(`   AI available for: ${response.data.questions.filter(q => q.aiAvailable).length} questions`);
      
      return response.data.questions;
    } catch (error: any) {
      console.error('❌ Failed to load questions:', error.response?.data || error.message);
      throw error;
    }
  }

  /**
   * Get details for a specific question/indicator
   */
  async getQuestionDetails(indicatorCode: string): Promise<QuestionDetailsResponse['question']> {
    try {
      const response = await this.apiClient.get<QuestionDetailsResponse>(`/questions/${indicatorCode}`);
      return response.data.question;
    } catch (error: any) {
      console.error(`❌ Failed to get question details for ${indicatorCode}:`, error.response?.data || error.message);
      throw error;
    }
  }

  /**
   * Get list of all indicators that have trained AI models
   */
  async getTrainedIndicators(): Promise<string[]> {
    try {
      const response = await this.apiClient.get<{
        success: boolean;
        total_trained: number;
        indicators: string[];
      }>('/indicators');
      
      console.log(`🤖 AI trained for ${response.data.total_trained} indicators`);
      return response.data.indicators;
    } catch (error: any) {
      console.error('❌ Failed to get trained indicators:', error.response?.data || error.message);
      throw error;
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
      console.log(`   Average confidence: ${response.data.statistics.average_confidence.toFixed(1)}%`);
      
      return response.data;
    } catch (error: any) {
      console.error('❌ AI prediction failed:', error.response?.data || error.message);
      throw error;
    }
  }

  /**
   * Predict with question metadata merged
   * Returns predictions with full question details
   */
  async predictWithMetadata(input: AssessmentInput): Promise<{
    predictions: Array<{
      indicator: string;
      value: string;
      confidence: number;
      question: Question;
    }>;
    statistics: PredictionResponse['statistics'];
  }> {
    try {
      // Load questions and predictions in parallel
      const [predictionsResponse, questions] = await Promise.all([
        this.predictAssessment(input),
        this.loadQuestions()
      ]);

      // Merge predictions with question metadata
      const mergedPredictions = Object.entries(predictionsResponse.predictions).map(
        ([indicator, prediction]) => ({
          indicator,
          value: prediction.value,
          confidence: prediction.confidence,
          question: this.questionsCache?.get(indicator) || {
            indicatorCode: indicator,
            description: 'Unknown',
            aiAvailable: true
          } as Question
        })
      );

      return {
        predictions: mergedPredictions,
        statistics: predictionsResponse.statistics
      };
    } catch (error: any) {
      console.error('❌ Predict with metadata failed:', error);
      throw error;
    }
  }

  /**
   * Batch predict multiple assessments
   */
  async predictBatch(assessments: AssessmentInput[]): Promise<any> {
    try {
      console.log(`🤖 Batch prediction for ${assessments.length} assessments...`);
      
      const response = await this.apiClient.post('/predict/batch', {
        assessments
      });
      
      console.log(`✅ Batch prediction complete: ${response.data.total} processed`);
      return response.data;
    } catch (error: any) {
      console.error('❌ Batch prediction failed:', error.response?.data || error.message);
      throw error;
    }
  }

  /**
   * Format AI predictions for form pre-filling
   * Only includes predictions above confidence threshold
   */
  formatPredictionsForForm(
    predictions: Record<string, PredictionResult>,
    minConfidence: number = 50
  ): Record<string, string> {
    const formData: Record<string, string> = {};
    
    Object.entries(predictions).forEach(([indicator, result]) => {
      if (result.confidence >= minConfidence) {
        formData[indicator] = result.value;
      }
    });
    
    return formData;
  }

  /**
   * Get questions filtered by category
   */
  async getQuestionsByCategory(categoryId: string): Promise<Question[]> {
    const allQuestions = await this.loadQuestions();
    return allQuestions.filter(q => q.categoryId === categoryId);
  }

  /**
   * Get only questions that have AI predictions available
   */
  async getAIEnabledQuestions(): Promise<Question[]> {
    const allQuestions = await this.loadQuestions();
    return allQuestions.filter(q => q.aiAvailable);
  }

  /**
   * Clear the questions cache
   */
  clearCache(): void {
    this.questionsCache = null;
    this.questionsCacheExpiry = 0;
    console.log('🗑️ Questions cache cleared');
  }
}

// Export singleton instance
export default new AssessmentAIService();
export { AssessmentAIService };
export type { 
  AssessmentInput, 
  PredictionResult, 
  PredictionResponse,
  Question,
  QuestionsResponse,
  QuestionDetailsResponse
};
