'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ChartBarIcon,
  CpuChipIcon,
  EyeIcon,
  ChatBubbleBottomCenterTextIcon,
  SparklesIcon,
  TrophyIcon,
  ClockIcon,
  ChartPieIcon
} from '@heroicons/react/24/outline';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import Link from 'next/link';

interface ModelData {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  type: string;
}

interface ModelPerformance {
  models: ModelData[];
}



export default function PerformancePage() {
  const [performance, setPerformance] = useState<ModelPerformance | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<'multimodal' | 'vision' | 'text'>('multimodal');
  
  // Use environment variable or fallback to Railway API URL
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://multimodale-commerceproductclassificationsys-production.up.railway.app';

  useEffect(() => {
    fetchPerformance();
  }, []);

  const fetchPerformance = async () => {
    try {
      const response = await fetch(`${API_URL}/api/models/performance`);
      const data = await response.json();
      setPerformance(data);
    } catch (error) {
      console.error('Error fetching performance:', error);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to get model by type
  const getModelByType = (type: string) => {
    return performance?.models.find(model => model.type === type);
  };

  // Helper function to safely get performance data
  const getPerformanceData = () => {
    if (!performance?.models) return null;
    
    const multimodal = getModelByType('multimodal');
    const vision = getModelByType('vision');
    const text = getModelByType('text');
    
    return { multimodal, vision, text };
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading performance data...</p>
        </div>
      </div>
    );
  }

  const performanceData = getPerformanceData();
  
  if (!performanceData || !performanceData.multimodal || !performanceData.vision || !performanceData.text) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600">Error loading performance data</p>
          <button 
            onClick={fetchPerformance}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Reload
          </button>
        </div>
      </div>
    );
  }

  const { multimodal, vision, text } = performanceData;

  const modelCards = [
    {
      type: 'multimodal' as const,
      name: multimodal.name,
      accuracy: multimodal.accuracy,
      f1: multimodal.f1_score,
      precision: multimodal.precision,
      recall: multimodal.recall,
      icon: SparklesIcon,
      color: 'from-purple-600 to-pink-600',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200'
    },
    {
      type: 'text' as const,
      name: text.name,
      accuracy: text.accuracy,
      f1: text.f1_score,
      precision: text.precision,
      recall: text.recall,
      icon: ChatBubbleBottomCenterTextIcon,
      color: 'from-blue-600 to-cyan-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200'
    },
    {
      type: 'vision' as const,
      name: vision.name,
      accuracy: vision.accuracy,
      f1: vision.f1_score,
      precision: vision.precision,
      recall: vision.recall,
      icon: EyeIcon,
      color: 'from-green-600 to-emerald-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200'
    }
  ];

  // Prepare chart data
  const comparisonData = [
    {
      name: 'Accuracy',
      Multimodal: multimodal.accuracy * 100,
      'Text Only': text.accuracy * 100,
      'Vision Only': vision.accuracy * 100,
    },
    {
      name: 'F1 Score',
      Multimodal: multimodal.f1_score * 100,
      'Text Only': text.f1_score * 100,
      'Vision Only': vision.f1_score * 100,
    },
    {
      name: 'Precision',
      Multimodal: multimodal.precision * 100,
      'Text Only': text.precision * 100,
      'Vision Only': vision.precision * 100,
    },
    {
      name: 'Recall',
      Multimodal: multimodal.recall * 100,
      'Text Only': text.recall * 100,
      'Vision Only': vision.recall * 100,
    }
  ];

  const radarData = [
    {
      metric: 'Accuracy',
      Multimodal: multimodal.accuracy * 100,
      'Text Only': text.accuracy * 100,
      'Vision Only': vision.accuracy * 100,
    },
    {
      metric: 'Precision',
      Multimodal: multimodal.precision * 100,
      'Text Only': text.precision * 100,
      'Vision Only': vision.precision * 100,
    },
    {
      metric: 'Recall',
      Multimodal: multimodal.recall * 100,
      'Text Only': text.recall * 100,
      'Vision Only': vision.recall * 100,
    },
    {
      metric: 'F1 Score',
      Multimodal: multimodal.f1_score * 100,
      'Text Only': text.f1_score * 100,
      'Vision Only': vision.f1_score * 100,
    }
  ];

  const categoryPerformance = [
    { name: 'Electronics', accuracy: 92, count: 15420 },
    { name: 'Home & Garden', accuracy: 88, count: 8930 },
    { name: 'Clothing', accuracy: 85, count: 12240 },
    { name: 'Sports', accuracy: 83, count: 6780 },
    { name: 'Books', accuracy: 90, count: 4560 },
    { name: 'Health', accuracy: 87, count: 3240 }
  ];

  const modelArchitectures = [
    {
      name: 'Computer Vision Models',
      models: ['ResNet50', 'ResNet101', 'DenseNet121', 'ConvNeXt V2', 'Vision Transformer', 'Swin Transformer'],
      accuracy: 76,
      color: '#10B981'
    },
    {
      name: 'NLP Models',
      models: ['MiniLM', 'BERT', 'OpenAI Embeddings'],
      accuracy: 82,
      color: '#3B82F6'
    },
    {
      name: 'Classical ML',
      models: ['Random Forest', 'Logistic Regression', 'SVM'],
      accuracy: 78,
      color: '#F59E0B'
    },
    {
      name: 'Deep Learning',
      models: ['MLP Networks', 'Custom Architectures'],
      accuracy: 84,
      color: '#8B5CF6'
    }
  ];

  // Custom tooltip formatter
  const customTooltipFormatter = (value: number) => [`${value.toFixed(1)}%`, ''];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Navigation */}
      <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link href="/" className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <ChartBarIcon className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Model Performance
              </h1>
            </Link>
            <Link 
              href="/"
              className="text-gray-600 hover:text-blue-600 transition-colors"
            >
              ← Back to Home
            </Link>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gray-800 mb-6">
            Performance <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Analytics</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Comprehensive analysis of model performance across different architectures and approaches
          </p>
        </motion.div>

        {/* Key Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12"
        >
          {[
            { label: 'Best Accuracy', value: '87%', icon: TrophyIcon, color: 'text-yellow-600' },
            { label: 'Models Trained', value: '10+', icon: CpuChipIcon, color: 'text-blue-600' },
            { label: 'Training Time', value: '48h', icon: ClockIcon, color: 'text-green-600' },
            { label: 'Categories', value: '100+', icon: ChartPieIcon, color: 'text-purple-600' }
          ].map((metric, index) => (
            <div key={index} className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 text-center border border-white/20">
              <metric.icon className={`w-8 h-8 ${metric.color} mx-auto mb-3`} />
              <div className="text-3xl font-bold text-gray-800 mb-1">{metric.value}</div>
              <div className="text-sm text-gray-600">{metric.label}</div>
            </div>
          ))}
        </motion.div>

        {/* Model Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 border border-white/20 mb-12"
        >
          <h2 className="text-3xl font-bold text-gray-800 mb-8">Model Comparison</h2>
          
          {loading ? (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <p className="text-gray-600 mt-4">Loading performance data...</p>
            </div>
          ) : (
            <div className="space-y-8">
              {/* Model Cards */}
              <div className="grid md:grid-cols-3 gap-6">
                {modelCards.map((model, index) => {
                  return (
                    <motion.div
                      key={model.type}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.6, delay: index * 0.1 }}
                      className={`${model.bgColor} rounded-2xl p-6 border ${model.borderColor} hover:shadow-lg transition-all duration-300 cursor-pointer ${
                        selectedModel === model.type ? 'ring-2 ring-blue-500' : ''
                      }`}
                      onClick={() => setSelectedModel(model.type)}
                    >
                      <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${model.color} flex items-center justify-center mb-4`}>
                        <model.icon className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-bold text-gray-800 mb-2">{model.name}</h3>
                      <p className="text-gray-600 text-sm mb-4">Type: {model.type}</p>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Accuracy:</span>
                          <span className="font-semibold text-gray-800">{(model.accuracy * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">F1 Score:</span>
                          <span className="font-semibold text-gray-800">{(model.f1 * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>

              {/* Performance Charts */}
              <div className="grid lg:grid-cols-2 gap-8">
                {/* Bar Chart */}
                <div className="bg-white/80 rounded-xl p-6 border border-gray-200">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">Metrics Comparison</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={comparisonData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip formatter={customTooltipFormatter} />
                      <Bar dataKey="Multimodal" fill="#8B5CF6" />
                      <Bar dataKey="Text Only" fill="#3B82F6" />
                      <Bar dataKey="Vision Only" fill="#10B981" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Radar Chart */}
                <div className="bg-white/80 rounded-xl p-6 border border-gray-200">
                  <h3 className="text-xl font-bold text-gray-800 mb-4">
                    All Models Performance
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="metric" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar
                        name="Multimodal"
                        dataKey="Multimodal"
                        stroke="#8B5CF6"
                        fill="#8B5CF6"
                        fillOpacity={0.3}
                      />
                      <Radar
                        name="Text Only"
                        dataKey="Text Only"
                        stroke="#3B82F6"
                        fill="#3B82F6"
                        fillOpacity={0.3}
                      />
                      <Radar
                        name="Vision Only"
                        dataKey="Vision Only"
                        stroke="#10B981"
                        fill="#10B981"
                        fillOpacity={0.3}
                      />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </motion.div>

        {/* Category Performance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 border border-white/20 mb-12"
        >
          <h2 className="text-3xl font-bold text-gray-800 mb-8">Performance by Category</h2>
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="bg-white/80 rounded-xl p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Accuracy by Category</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={categoryPerformance} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip formatter={(value: number) => [`${value}%`, 'Accuracy']} />
                  <Bar dataKey="accuracy" fill="#3B82F6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-xl font-bold text-gray-800">Category Statistics</h3>
              {categoryPerformance.map((category, index) => (
                <div key={index} className="bg-white/80 rounded-lg p-4 border border-gray-200">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-gray-800">{category.name}</span>
                    <span className="text-2xl font-bold text-blue-600">{category.accuracy}%</span>
                  </div>
                  <div className="text-sm text-gray-600 mb-2">
                    {category.count.toLocaleString()} products
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                      style={{ width: `${category.accuracy}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Architecture Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 border border-white/20"
        >
          <h2 className="text-3xl font-bold text-gray-800 mb-8">Architecture Performance</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {modelArchitectures.map((arch, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/80 rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-all duration-300"
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold text-gray-800">{arch.name}</h3>
                  <div className="text-2xl font-bold" style={{ color: arch.color }}>
                    {arch.accuracy}%
                  </div>
                </div>
                <div className="space-y-2 mb-4">
                  {arch.models.map((model, idx) => (
                    <div key={idx} className="text-sm text-gray-600 bg-gray-50 rounded px-2 py-1">
                      {model}
                    </div>
                  ))}
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 rounded-full"
                    style={{ 
                      width: `${arch.accuracy}%`,
                      backgroundColor: arch.color
                    }}
                  ></div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
} 