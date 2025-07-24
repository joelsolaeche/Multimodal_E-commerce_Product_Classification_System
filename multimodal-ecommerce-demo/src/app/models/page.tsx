'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  CpuChipIcon,
  EyeIcon,
  ChatBubbleBottomCenterTextIcon,
  SparklesIcon,
  ChartBarIcon,
  BeakerIcon,
  CubeIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';
import Link from 'next/link';

interface ModelInfo {
  id: string;
  name: string;
  category: 'vision' | 'nlp' | 'multimodal' | 'classical';
  accuracy: string;
  parameters: string;
  description: string;
  architecture: string;
  strengths: string[];
  icon: React.ComponentType<{ className?: string }>;
  color: string;
}

export default function ModelsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const models: ModelInfo[] = [
    // Vision Models
    {
      id: 'resnet50',
      name: 'ResNet-50',
      category: 'vision',
      accuracy: '82.3%',
      parameters: '25.6M',
      description: 'Deep residual network with skip connections enabling training of very deep networks',
      architecture: 'Convolutional Neural Network with residual blocks',
      strengths: ['Skip connections prevent vanishing gradients', 'Excellent feature extraction', 'Well-established architecture'],
      icon: EyeIcon,
      color: 'from-blue-500 to-cyan-500'
    },
    {
      id: 'densenet121',
      name: 'DenseNet-121',
      category: 'vision',
      accuracy: '83.1%',
      parameters: '7.9M',
      description: 'Dense connectivity pattern where each layer receives feature maps from all preceding layers',
      architecture: 'Densely connected convolutional networks',
      strengths: ['Parameter efficient', 'Strong feature reuse', 'Reduced overfitting'],
      icon: EyeIcon,
      color: 'from-green-500 to-emerald-500'
    },
    {
      id: 'convnext',
      name: 'ConvNeXt V2',
      category: 'vision',
      accuracy: '84.7%',
      parameters: '28.6M',
      description: 'Modern ConvNet design inspired by Vision Transformers with improved training strategies',
      architecture: 'Modernized convolutional architecture',
      strengths: ['State-of-the-art CNN performance', 'Improved training stability', 'Better generalization'],
      icon: EyeIcon,
      color: 'from-purple-500 to-pink-500'
    },
    {
      id: 'vit',
      name: 'Vision Transformer',
      category: 'vision',
      accuracy: '83.8%',
      parameters: '86.6M',
      description: 'Pure transformer architecture applied to image classification with patch-based processing',
      architecture: 'Transformer encoder with image patches as tokens',
      strengths: ['Attention-based processing', 'Global context modeling', 'Scalable architecture'],
      icon: CubeIcon,
      color: 'from-indigo-500 to-purple-500'
    },
    {
      id: 'swin',
      name: 'Swin Transformer',
      category: 'vision',
      accuracy: '84.2%',
      parameters: '28.3M',
      description: 'Hierarchical vision transformer with shifted windowing for efficient computation',
      architecture: 'Hierarchical transformer with shifted windows',
      strengths: ['Linear computational complexity', 'Hierarchical representations', 'Cross-window connections'],
      icon: CubeIcon,
      color: 'from-orange-500 to-red-500'
    },
    // NLP Models
    {
      id: 'bert-base',
      name: 'BERT Base',
      category: 'nlp',
      accuracy: '79.2%',
      parameters: '110M',
      description: 'Bidirectional encoder representations from transformers for language understanding',
      architecture: 'Bidirectional transformer encoder',
      strengths: ['Bidirectional context', 'Pre-trained representations', 'Fine-tuning capability'],
      icon: ChatBubbleBottomCenterTextIcon,
      color: 'from-teal-500 to-cyan-500'
    },
    {
      id: 'bert-minilm',
      name: 'BERT MiniLM-L12',
      category: 'nlp',
      accuracy: '78.8%',
      parameters: '33M',
      description: 'Distilled BERT model with reduced parameters while maintaining performance',
      architecture: 'Distilled transformer encoder',
      strengths: ['Compact model size', 'Fast inference', 'Good performance retention'],
      icon: ChatBubbleBottomCenterTextIcon,
      color: 'from-emerald-500 to-teal-500'
    },
    {
      id: 'roberta',
      name: 'RoBERTa Base',
      category: 'nlp',
      accuracy: '79.6%',
      parameters: '125M',
      description: 'Robustly optimized BERT with improved training methodology',
      architecture: 'Optimized transformer encoder',
      strengths: ['Improved training strategy', 'Better performance', 'Robust representations'],
      icon: ChatBubbleBottomCenterTextIcon,
      color: 'from-blue-500 to-indigo-500'
    },
    // Multimodal Models
    {
      id: 'early-fusion',
      name: 'Early Fusion',
      category: 'multimodal',
      accuracy: '83.9%',
      parameters: 'Variable',
      description: 'Concatenates image and text features early in the processing pipeline',
      architecture: 'Feature concatenation + MLP classifier',
      strengths: ['Simple implementation', 'Joint feature learning', 'Good baseline performance'],
      icon: SparklesIcon,
      color: 'from-violet-500 to-purple-500'
    },
    {
      id: 'late-fusion',
      name: 'Late Fusion',
      category: 'multimodal',
      accuracy: '84.1%',
      parameters: 'Variable',
      description: 'Combines predictions from separate image and text models',
      architecture: 'Separate encoders + prediction fusion',
      strengths: ['Modality-specific optimization', 'Interpretable decisions', 'Flexible weighting'],
      icon: SparklesIcon,
      color: 'from-pink-500 to-rose-500'
    },
    {
      id: 'attention-fusion',
      name: 'Attention Fusion',
      category: 'multimodal',
      accuracy: '85.2%',
      parameters: 'Variable',
      description: 'Uses learned attention weights to optimally combine multimodal features',
      architecture: 'Cross-modal attention mechanism',
      strengths: ['Optimal feature weighting', 'Best performance', 'Adaptive fusion'],
      icon: SparklesIcon,
      color: 'from-amber-500 to-orange-500'
    },
    // Classical ML
    {
      id: 'random-forest',
      name: 'Random Forest',
      category: 'classical',
      accuracy: '76.4%',
      parameters: '~1K trees',
      description: 'Ensemble of decision trees with random feature selection',
      architecture: 'Ensemble of decision trees',
      strengths: ['Interpretable results', 'Handles mixed data types', 'Robust to overfitting'],
      icon: BeakerIcon,
      color: 'from-green-600 to-lime-500'
    }
  ];

  const categories = [
    { id: 'all', name: 'All Models', icon: CpuChipIcon, count: models.length },
    { id: 'vision', name: 'Computer Vision', icon: EyeIcon, count: models.filter(m => m.category === 'vision').length },
    { id: 'nlp', name: 'Natural Language', icon: ChatBubbleBottomCenterTextIcon, count: models.filter(m => m.category === 'nlp').length },
    { id: 'multimodal', name: 'Multimodal', icon: SparklesIcon, count: models.filter(m => m.category === 'multimodal').length },
    { id: 'classical', name: 'Classical ML', icon: BeakerIcon, count: models.filter(m => m.category === 'classical').length }
  ];

  const filteredModels = selectedCategory === 'all' 
    ? models 
    : models.filter(model => model.category === selectedCategory);

  const bestModel = models.reduce((best, current) => 
    parseFloat(current.accuracy) > parseFloat(best.accuracy) ? current : best
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <Link href="/" className="text-blue-400 hover:text-blue-300 mb-2 inline-block">
                ← Back to Home
              </Link>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                ML Model Architecture
              </h1>
              <p className="text-gray-400 mt-2">
                Comprehensive overview of {models.length} machine learning models and their performance
              </p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-400">Best Performing Model</div>
              <div className="text-2xl font-bold text-green-400">{bestModel.name}</div>
              <div className="text-lg text-white">{bestModel.accuracy} accuracy</div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Category Filter */}
        <div className="mb-8">
          <div className="flex flex-wrap gap-4">
            {categories.map((category) => {
              const IconComponent = category.icon;
              return (
                <button
                  key={category.id}
                  onClick={() => setSelectedCategory(category.id)}
                  className={`px-6 py-3 rounded-xl font-medium transition-all duration-200 flex items-center space-x-2 ${
                    selectedCategory === category.id
                      ? 'bg-blue-600 text-white shadow-lg'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  <IconComponent className="h-5 w-5" />
                  <span>{category.name}</span>
                  <span className="bg-gray-700 text-gray-300 px-2 py-1 rounded-full text-xs">
                    {category.count}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Models Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {filteredModels.map((model, index) => {
            const IconComponent = model.icon;
            return (
              <motion.div
                key={model.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-gray-600 transition-all duration-200"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`p-3 rounded-lg bg-gradient-to-r ${model.color}`}>
                      <IconComponent className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white">{model.name}</h3>
                      <p className="text-sm text-gray-400 capitalize">{model.category.replace('_', ' ')}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-green-400">{model.accuracy}</div>
                    <div className="text-xs text-gray-400">Accuracy</div>
                  </div>
                </div>

                {/* Description */}
                <p className="text-gray-300 mb-4 leading-relaxed">{model.description}</p>

                {/* Architecture & Parameters */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                  <div className="bg-gray-700/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">Architecture</div>
                    <div className="text-sm text-white font-medium">{model.architecture}</div>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">Parameters</div>
                    <div className="text-sm text-white font-medium">{model.parameters}</div>
                  </div>
                </div>

                {/* Strengths */}
                <div>
                  <h4 className="text-sm font-semibold text-gray-300 mb-2">Key Strengths</h4>
                  <div className="space-y-1">
                    {model.strengths.map((strength, i) => (
                      <div key={i} className="flex items-center text-sm text-gray-400">
                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full mr-2 flex-shrink-0"></div>
                        {strength}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Best Model Badge */}
                {model.id === bestModel.id && (
                  <div className="mt-4 inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-yellow-500 to-orange-500 text-white">
                    <ChartBarIcon className="h-3 w-3 mr-1" />
                    Best Performance
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Performance Summary */}
        <div className="mt-12 bg-gradient-to-br from-blue-900/50 to-purple-900/50 rounded-xl p-8 border border-blue-500/30">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
            <ChartBarIcon className="h-6 w-6 mr-2 text-blue-400" />
            Performance Summary
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-green-400 mb-2">85.2%</div>
              <div className="text-gray-300">Best Accuracy</div>
              <div className="text-sm text-gray-400">Attention Fusion</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-400 mb-2">{models.length}</div>
              <div className="text-gray-300">Models Trained</div>
              <div className="text-sm text-gray-400">Multiple Architectures</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-400 mb-2">200+</div>
              <div className="text-gray-300">Training Hours</div>
              <div className="text-sm text-gray-400">GPU Compute Time</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-yellow-400 mb-2">3</div>
              <div className="text-gray-300">Modalities</div>
              <div className="text-sm text-gray-400">Vision + Text + Fusion</div>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="mt-12 text-center">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Link href="/demo" className="group">
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-blue-500/50 transition-all duration-300 hover:bg-gray-700/50">
                <SparklesIcon className="h-8 w-8 text-blue-400 mx-auto mb-3" />
                <h3 className="text-lg font-semibold text-white mb-2">Try Interactive Demo</h3>
                <p className="text-gray-400 text-sm">Test these models with your own data</p>
              </div>
            </Link>

            <Link href="/performance" className="group">
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-green-500/50 transition-all duration-300 hover:bg-gray-700/50">
                <ChartBarIcon className="h-8 w-8 text-green-400 mx-auto mb-3" />
                <h3 className="text-lg font-semibold text-white mb-2">View Performance</h3>
                <p className="text-gray-400 text-sm">Detailed metrics and comparisons</p>
              </div>
            </Link>

            <Link href="/" className="group">
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-purple-500/50 transition-all duration-300 hover:bg-gray-700/50">
                <DocumentTextIcon className="h-8 w-8 text-purple-400 mx-auto mb-3" />
                <h3 className="text-lg font-semibold text-white mb-2">Learn More</h3>
                <p className="text-gray-400 text-sm">Technical documentation</p>
              </div>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
} 